from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def _as_bf(t, want_f=4):
    """Devuelve tensor float32 de forma (B, F). Acomoda dims, recorta/padea a F=4."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    t = t.float()
    # Quitar dims singleton intermedias
    if t.ndim == 3:  # (B,1,F) o (1,1,F)
        t = t.squeeze(1)
    elif t.ndim == 1:  # (F,)
        t = t.unsqueeze(0)  # (1,F)
    elif t.ndim == 2:
        pass  # (B,F)
    else:
        # cualquier otra cosa → aplanar y dejar como (1,F)
        t = t.reshape(1, -1)

    # Ajustar F
    f = t.shape[-1]
    if f > want_f:
        t = t[..., :want_f]
    elif f < want_f:
        pad = torch.zeros(t.shape[:-1] + (want_f - f,), dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad], dim=-1)
    return t  # (B,F)


class TinyMixer(nn.Module):
    """
    Modelo liviano para demo. Concatena x(4) + fb(4) → 8 → MLP.
    Salidas:
      - props (softmax 4): proporciones P,T,M,C
      - perf  (relu 3)    : [E_kWh/kg, W_L/kg, Crew_min/kg]
    """
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.head_props = nn.Linear(8, 4)
        self.head_perf  = nn.Linear(8, 3)

    def forward(self, x, fb):
        x  = _as_bf(x, 4)   # (B,4)
        fb = _as_bf(fb, 4)  # (B,4)

        # Alinear batch: si uno es B=1 y otro B>1, expandimos
        bx, bf = x.shape[0], fb.shape[0]
        if bx != bf:
            if bx == 1:
                x = x.expand(bf, -1)
            elif bf == 1:
                fb = fb.expand(bx, -1)
            else:
                raise RuntimeError(f"Batch mismatch x={bx}, fb={bf}")

        h = torch.cat([x, fb], dim=-1)  # (B,8)
        h = self.backbone(h)
        props = F.softmax(self.head_props(h), dim=-1)
        perf  = F.relu(self.head_perf(h))
        return props, perf


def train_rexai_2025(data, epochs: int = 5):
    # Para demo no entrenamos; retornamos el modelo listo
    return TinyMixer()
