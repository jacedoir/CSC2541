import torch
import torch.nn as nn
import torch.optim as optim
from mobile_sam import sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download
from UniRepLKNetSmall import UniRepLKNetSmall

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# === Define SAM Encoder ===
class SAMEncoder(nn.Module):
    def __init__(self, device):
        super(SAMEncoder, self).__init__()
        self.model_type = "vit_t"
        self.sam_checkpoint = "./weights/mobile_sam.pt"

        self.device = device

        self.mobile_sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.mobile_sam.to(device=self.device)
        self.mobile_sam.eval()

        self.predictor = SamPredictor(self.mobile_sam)

    def forward(self, x):
        """
        pixel value of x is in [0, 255]
        """
        self.predictor.set_image(x)
        return self.predictor.features #torch.Size([1, 256, 64, 64])

# === Define G Encoder (UniRepLKNet) ===
class GEncoder(nn.Module):
    def __init__(self, device):
        self.device = device
        self.model = UniRepLKNetSmall(in_channels=3, base_channels=96)
        self.model.to(device=self.device)
        
    def forward(self, x):
        dilated_outputs, final_output = self.model(x)
        return dilated_outputs, final_output

class BSAModule(nn.Module):
    def __init__(self, channels):
        super(BSAModule, self).__init__()
        
        # Layers for splitting and dilated convolutions
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, dilation=d, padding=d)
            for d in [1, 2, 3, 5]
        ])
        self.post_conv = nn.Conv2d(4 * channels, channels, kernel_size=1)

    def forward(self, xG, xD, xS):
        # Split each input into four parts
        parts_xG = torch.chunk(xG, 4, dim=1)
        parts_xD = torch.chunk(xD, 4, dim=1)
        parts_xS = torch.chunk(xS, 4, dim=1)
        
        # Concatenate and apply dilated convolutions
        mixed_features = []
        for j in range(4):
            combined = torch.cat([parts_xG[j], parts_xD[j], parts_xS[j]], dim=1)
            conv_out = self.dilated_convs[j](combined)
            mixed_features.append(conv_out)
        
        # Concatenate all features in channel direction and reduce dimensions
        F = torch.cat(mixed_features, dim=1)
        return self.post_conv(F)

class CNNDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNDeconvBlock, self).__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=d, padding=d)
            for d in [1, 2, 3, 5]
        ])
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        features = [conv(x) for conv in self.dilated_convs]
        fused_features = torch.cat(features, dim=1)
        upsampled = self.upsample(fused_features)
        return self.relu(self.bn(upsampled))

class GDDecoder(nn.Module):
    def __init__(self, base_channels):
        super(GDDecoder, self).__init__()
        
        # BSAM and upsampling modules for each level of decoding
        self.bsam1 = BSAModule(base_channels * 8)
        self.bsam2 = BSAModule(base_channels * 4)
        self.bsam3 = BSAModule(base_channels * 2)
        self.bsam4 = BSAModule(base_channels)
        
        # CNN & Deconv Blocks
        self.deconv_block1 = CNNDeconvBlock(base_channels * 8, base_channels * 4)
        self.deconv_block2 = CNNDeconvBlock(base_channels * 4, base_channels * 2)
        self.deconv_block3 = CNNDeconvBlock(base_channels * 2, base_channels)
        self.deconv_block4 = CNNDeconvBlock(base_channels, base_channels)

    def forward(self, xG_features, xS):
        # Assume xG_features is a list of features from the G encoder (largest to smallest scale)
        xG5, xG4, xG3, xG2 = xG_features
        
        # Initial processing through BSAM and upsampling blocks
        xD4 = self.bsam1(xG5, xG5, xS)
        xD4 = self.deconv_block1(xD4)
        
        xD3 = self.bsam2(xG4, xD4, xS)
        xD3 = self.deconv_block2(xD3)
        
        xD2 = self.bsam3(xG3, xD3, xS)
        xD2 = self.deconv_block3(xD2)
        
        xD1 = self.bsam4(xG2, xD2, xS)
        y_hat = self.deconv_block4(xD1)
        
        return y_hat


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels, base_channels=64):
        super(PatchGANDiscriminator, self).__init__()
        
        # Define layers of PatchGAN
        layers = [
            # First layer (no BatchNorm)
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Subsequent layers with increasing channels
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final output layer
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class LossFunctions(nn.Module):
    def __init__(self, alpha=0.5, beta=25, lambda_gan=1, lambda_sac=10, lambda_cl=10, tau=0.07):
        super(LossFunctions, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_gan = lambda_gan
        self.lambda_sac = lambda_sac
        self.lambda_cl = lambda_cl
        self.tau = tau

    def structural_consistency_loss(self, features_x, features_y):
        # Calculate structural consistency using cosine distance
        Sx, Sy = features_x, features_y
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        structural_loss = 1 - cos_sim(Sx, Sy).mean()
        return structural_loss

    def appearance_consistency_loss(self, generated_features, real_features):
        # Sort the features
        f_generated_sorted, _ = torch.sort(generated_features.view(-1))
        f_real_sorted, _ = torch.sort(real_features.view(-1))

        # EFDM Loss
        loss = torch.mean((f_generated_sorted - f_real_sorted) ** 2)
        return loss

    def patch_nce_loss(self, q, k_pos, k_neg):
        # PatchNCE Loss calculation
        pos_term = torch.exp(torch.dot(q, k_pos) / self.tau)
        neg_term = torch.sum(torch.exp(torch.dot(q, k_neg) / self.tau))

        loss = -torch.log(pos_term / (pos_term + neg_term))
        return loss

    def forward(self, x, y, generated_y, real_features):
        # Structural Appearance Consistency Loss
        features_x = self.extract_features(x)  # Assume some method to extract features from the visible image
        features_generated_y = self.extract_features(generated_y)  # Features from generated IR image
        features_y = self.extract_features(y)  # Features from real IR image
        
        sac_loss = self.alpha * self.structural_consistency_loss(features_x, features_generated_y) + \
                   self.beta * self.appearance_consistency_loss(features_generated_y, features_y)
        
        # LSGAN Loss (Assuming some discriminator D is already defined and you have real and generated images)
        gan_loss = self.lsgan_loss(real_features, generated_y)  # Placeholder for LSGAN loss computation

        # PatchNCE loss can be integrated as needed. Here is a simple call assuming q, k+, k- are given.
        # You can modify this part according to how you get your query and keys.
        # Example:
        # patch_nce_loss = self.patch_nce_loss(q, k_pos, k_neg)

        # Combine losses
        total_loss = self.lambda_gan * gan_loss + self.lambda_sac * sac_loss  # + self.lambda_cl * patch_nce_loss
        return total_loss

    def lsgan_loss(self, real, fake):
        # Define LSGAN Loss
        return (torch.mean((real - 1) ** 2) + torch.mean(fake ** 2)) / 2

    def extract_features(self, x):
        # Placeholder for feature extraction logic using a model (e.g., VGG16, SAM, etc.)
        # Here you would call your encoder model to get features
        return x  # Replace with actual feature extraction

# === Full AerialIRGAN Model ===
class AerialIRGAN(nn.Module):
    def __init__(self):
        super(AerialIRGAN, self).__init__()
        self.sam_encoder = SAMEncoder(device)
        self.g_encoder = GEncoder(device)
        self.decoder = GDDecoder(device)

    def forward(self, x):
        x_sam = self.sam_encoder(x)
        list_dilated_outputs, x_gen = self.g_encoder(x)
        out = self.decoder(list_dilated_outputs, x_sam)
        return out

# === Training Loop ===
def train():
    generator = AerialIRGAN()
    discriminator = PatchGANDiscriminator()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    sac_loss = LossFunctions()
    
    for epoch in range(epochs):
        for real_images in dataloader:
            # Generator forward pass
            gen_images = generator(real_images)
            
            # GAN loss
            d_loss, g_loss = compute_gan_loss(discriminator, real_images, gen_images)
            sac = sac_loss(real_images, gen_images)
            
            # Backpropagation for Generator
            g_total_loss = g_loss + sac
            optimizer_G.zero_grad()
            g_total_loss.backward()
            optimizer_G.step()
            
            # Backpropagation for Discriminator
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

# Run the training process
train()
