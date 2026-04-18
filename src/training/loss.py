import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class EdgeAwareSharpnessLoss(nn.Module):
    """
    Custom penalty applied to gradients indicating anti-aliasing or smooth transitions.
    Forces the GAN to output hard, binary edges characteristic of true pixel art geometry.
    """
    def __init__(self, threshold=0.3, lower_bound=0.05):
        super(EdgeAwareSharpnessLoss, self).__init__()
        self.threshold = threshold
        self.lower_bound = lower_bound
        # Setup Sobel filters to detect edges
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, hr_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate penalty for smooth color transitions (anti-aliasing).
        
        Args:
            hr_pred (Tensor): Generated high-resolution image (B, 3, H, W) assumed to be in [0, 1].
            
        Returns:
            Tensor: Scalar loss value penalizing smooth transitions.
        """
        # Convert to grayscale to evaluate structural gradients
        gray = 0.299 * hr_pred[:, 0:1, :, :] + 0.587 * hr_pred[:, 1:2, :, :] + 0.114 * hr_pred[:, 2:3, :, :]
        
        # Calculate gradients using grouped convolutions to match dimensions
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Gradient magnitude
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
        # We penalize gradient magnitudes that fall in the "smooth" / anti-aliased range
        # E.g., magnitudes strictly between lower_bound and threshold
        # penalty = max(0, magnitude - lower_bound) * max(0, threshold - magnitude)
        # This formula peaks exactly halfway between lower_bound and threshold.
        penalty = F.relu(magnitude - self.lower_bound) * F.relu(self.threshold - magnitude)
        
        return penalty.mean()

class PaletteConstraintLoss(nn.Module):
    """
    Forces the generated colors to snap to a limited discrete color palette,
    which is characteristic of retro games (e.g., 8-bit or 16-bit color depth).
    """
    def __init__(self, color_depth=32.0):
        super(PaletteConstraintLoss, self).__init__()
        self.color_depth = color_depth

    def forward(self, hr_pred: torch.Tensor) -> torch.Tensor:
        """
        Penalize colors that do not fall exactly on the discretized color grid.
        Args:
            hr_pred (Tensor): Generated image in [0, 1].
        """
        # Scale to color grid
        scaled = hr_pred * self.color_depth
        
        # Distance to the nearest valid color grid point
        # This pushes the network to output flat, exact colors instead of continuous gradients.
        distance_to_grid = torch.abs(scaled - torch.round(scaled))
        
        return distance_to_grid.mean()


class GeneratorLoss(nn.Module):
    """
    Combined loss for the ESRGAN Generator.
    Consists of:
    1. L1 Pixel Loss (Content Loss)
    2. Perceptual Loss (LPIPS/VGG)
    3. Adversarial Loss (BCE with Logits for Discriminator output)
    4. Custom Edge-Aware Sharpness Loss (for Pixel Art constraints)
    """
    def __init__(self, pixel_weight=1e-2, perceptual_weight=1.0, adv_weight=5e-3, edge_weight=1e-1, palette_weight=5e-2):
        super(GeneratorLoss, self).__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adv_weight = adv_weight
        self.edge_weight = edge_weight
        self.palette_weight = palette_weight
        
        # Learned Perceptual Image Patch Similarity (LPIPS) loaded from requirements
        # Used because standard pixel loss (L1/MSE) creates blurry images
        self.perceptual_loss = lpips.LPIPS(net='vgg')
        
        self.edge_loss = EdgeAwareSharpnessLoss()
        self.palette_loss = PaletteConstraintLoss()

    def forward(self, hr_pred: torch.Tensor, hr_target: torch.Tensor, discriminator_pred: torch.Tensor) -> tuple:
        """
        Args:
            hr_pred: The output of the RRDBNet Generator
            hr_target: The actual real high-resolution sprite
            discriminator_pred: The Discriminator's logits output for hr_pred
        Returns:
            total_loss (Tensor), loss_dict (Dict tracking components)
        """
        device = hr_pred.device
        
        # 1. Pixel Loss (L1)
        pixel_loss = F.l1_loss(hr_pred, hr_target) * self.pixel_weight
        
        # 2. Perceptual Loss (LPIPS takes inputs in range [-1, 1], assuming inputs are [0, 1])
        lpips_pred = (hr_pred * 2) - 1
        lpips_target = (hr_target * 2) - 1
        # LPIPS returns spatial maps, so we mean them
        perc_loss = self.perceptual_loss(lpips_pred, lpips_target).mean() * self.perceptual_weight
        
        # 3. Adversarial Loss (Non-saturating Generator objective max log(D(G(z))))
        # Treat the discriminator as returning un-normalized logits
        target_real = torch.ones_like(discriminator_pred, device=device)
        adv_loss = F.binary_cross_entropy_with_logits(discriminator_pred, target_real) * self.adv_weight
        
        # 4. Custom Pixel-Art Edge-Aware Loss
        edge_loss = self.edge_loss(hr_pred) * self.edge_weight
        
        # 5. Palette Constraint Loss
        palette_loss = self.palette_loss(hr_pred) * self.palette_weight
        
        # Sum total
        total_loss = pixel_loss + perc_loss + adv_loss + edge_loss + palette_loss
        
        loss_dict = {
            "l1": pixel_loss.item(),
            "perceptual": perc_loss.item(),
            "adversarial": adv_loss.item(),
            "edge": edge_loss.item(),
            "palette": palette_loss.item(),
            "total": total_loss.item()
        }
        
        return total_loss, loss_dict
