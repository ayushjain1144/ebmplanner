import torch
from torch import nn


class RelationEBM(nn.Module):
    """Concept EBM for arbitrary relations."""

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.g_net = nn.Sequential(
            nn.Linear(8, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, inputs):
        """
        Forward pass.

        Inputs:
            sbox_centers (tensor): centers of subject boxes (B, N_rel, 2)
            sbox_sizes (tensor): sizes of subject boxes (B, N_rel, 2)
            obox_centers (tensor): centers of object boxes (B, N_rel, 2)
            obox_sizes (tensor): sizes of object boxes (B, N_rel, 2)
            _
        """
        if len(inputs) == 5:
            sbox_centers, sbox_sizes, obox_centers, obox_sizes, _ = inputs
        else:
            sbox_centers, sbox_sizes, obox_centers, obox_sizes = inputs
        # Embed object boxes to feature vectors
        subjs = torch.cat((
            sbox_centers - sbox_sizes / 2,
            sbox_centers + sbox_sizes / 2
        ), -1)
        objs = torch.cat((
            obox_centers - obox_sizes / 2,
            obox_centers + obox_sizes / 2
        ), -1)
        feats = torch.cat((
            subjs - objs,
            subjs - objs[..., (2, 3, 0, 1)]
        ), -1)
        # Compute energy
        return self.g_net(feats)


class ShapeEBM(nn.Module):
    """Concept EBM for shapes."""

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.f_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
        )
        layer = nn.TransformerEncoderLayer(128, 4, 128, batch_first=True)
        self.context = nn.TransformerEncoder(layer, 4)
        self.g_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, inputs):
        """
        Forward pass.

        Inputs:
            sbox_centers (tensor): centers of subject boxes (B, N, 3)
            sbox_sizes (unused)
            relations (unused)
            mask (tensor): (B, N), 1 if real object, 0 if padding
        """
        sbox_centers, _, _, mask = inputs
        sbox_centers = sbox_centers - sbox_centers.mean(1)[:, None]
        mask = ~mask.bool()
        # Embed object boxes to feature vectors
        feats = self.f_net(sbox_centers)
        # Contextualize, (B, 128)
        feats = self.context(feats, src_key_padding_mask=mask).mean(1)
        # Compute energy
        return self.g_net(feats)


class RelationEBM3D(nn.Module):
    """Concept EBM for arbitrary 3D relations."""

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.g_net = nn.Sequential(
            nn.Linear(12, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, inputs):
        """
        Forward pass.

        Inputs:
            sbox_centers (tensor): centers of subject boxes (B, N_rel, 3)
            sbox_sizes (tensor): sizes of subject boxes (B, N_rel, 3)
            obox_centers (tensor): centers of object boxes (B, N_rel, 3)
            obox_sizes (tensor): sizes of object boxes (B, N_rel, 3)
            rels (tensor): relation ids, (B, N_rel)
        """
        if len(inputs) == 5:
            sbox_centers, sbox_sizes, obox_centers, obox_sizes, _ = inputs
        else:
            sbox_centers, sbox_sizes, obox_centers, obox_sizes = inputs
        # Embed object boxes to feature vectors
        subjs = torch.cat((
            sbox_centers - sbox_sizes / 2,
            sbox_centers + sbox_sizes / 2
        ), -1)
        objs = torch.cat((
            obox_centers - obox_sizes / 2,
            obox_centers + obox_sizes / 2
        ), -1)
        feats = torch.cat((
            subjs - objs,
            subjs - objs[..., (3, 4, 5, 0, 1, 2)]
        ), -1)
        # Compute energy
        return self.g_net(feats)


class PosedShapeEBM(nn.Module):
    """Concept EBM for shapes of posed objects."""

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.f_net = nn.Sequential(
            nn.Linear(4, 128),
            nn.LeakyReLU()
        )
        layer = nn.TransformerEncoderLayer(128, 4, 128, batch_first=True)
        self.context = nn.TransformerEncoder(layer, 4)
        self.g_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, inputs):
        """
        Forward pass.

        Inputs:
            centers (tensor): centers of subject boxes (B, N, 2)
            thetas (tensor): orientations of subject boxes (B, N, 1)
        """
        points, thetas = inputs
        points = points - points.mean(1)[:, None]
        points = torch.cat((
            points,
            points + torch.stack((torch.cos(thetas), torch.sin(thetas)), -1)
        ), -1)
        # Embed object boxes to feature vectors
        feats = self.f_net(points)
        # Contextualize, (B, 128)
        feats = self.context(feats).mean(1)
        # Compute energy
        return self.g_net(feats)
