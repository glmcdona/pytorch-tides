
import torch
import math
import torch
from torch import nn
import torch.nn.functional as F
from .astro_cache import AstroCache

def ConstituentWeightBlock(in_features, mid_features, num_layers, *args, **kwargs):
  # Create sequential linear layers with activations
  layers = [nn.Linear(in_features, mid_features)]
  for _ in range(num_layers - 1):
    layers.append(nn.Linear(mid_features, mid_features))
    layers.append(nn.ReLU())

  layers.append(nn.Linear(mid_features, 1))

  # Create sequential block
  return nn.Sequential(*layers)

class TideNetGlobal(nn.Module):
  """
  Astronomical tidal model for tidal data coming in from various global locations. Used to fit a more
  complex tidal model from globally-distributed tidal data. For example Jason-3 satellite data.
  """
  def __init__(self, layer_width=30, layer_count=3):
    """
    Initialize the TideNetGlobal model for predicting global tides by latitude and longitude.

    Args:
        layer_width (int, optional): Width of each layer in the model. Defaults to 30.
        layer_count (int, optional): Number of layers in the model. Defaults to 3.
    """
    super().__init__()

    # Astro data for the specific times for all constituents
    self.astro = AstroCache(verbose=False)
    self.num_constituents = self.astro.num_constituents

    # Create the learned bias weight
    self.bias = nn.Parameter(torch.zeros(1))

    # Create the learned constituent amplitudes and phases
    self.layers_H = nn.ModuleList([]) # H, constituent amplitude by lat, lon
    self.layers_p = nn.ModuleList([]) # p, constituent phase by lat, lon
    for i in range(self.num_constituents):
      # constituent amplitude by lat, lon
      self.layers_H.append(ConstituentWeightBlock(
        in_features=2, mid_features=layer_width, num_layers=layer_count, bias=True))

      # constituent phase by lat, lon
      self.layers_p.append(ConstituentWeightBlock(
        in_features=2, mid_features=layer_width, num_layers=layer_count, bias=True))
  
  def cache_times(self, times):
    """
    Cache the astronomical data for the specific times for all constituents.
    """
    self.astro.add_times_to_astro(times)

  def forward(self, x, t):
    """
    Forward pass of the TideNetGlobal model.

    Args:
        x (_type_): Batch input data of latitude and longitudes. Shape: [batch_size, 2]
        t (_type_): Batime times input in seconds since epoch, UTC. Shape: [batch_size, 1]

    Returns:
        [float]: Height of the tide at the input latitude and longitude. Shape: [batch_size, 1]
    """

    # Height from a single constituent index n at time t is calculated as:
    #   h = H * f * cos( V + a * t + u - p )
    #  where:
    #    f, V, a, u: Astronomical data values at the specific time from self.astro.
    #    t: Hours since epoch
    #    H, p: Constituent amplitude and phase we fit with PyTorch. Varies with
    #      lat, lon and dependent on local geographic topography.
    #
    # Height from all constituents are added together to get the total height.
    # See https://github.com/sam-cox/pytides/wiki/Theory-of-the-Harmonic-Model-of-Tides

    # Load the astronomical data for the specific times for all constituents
    (a, u, f, V, t_hours_offset) = self.astro.get_astro(t.tolist())

    # Convert constants at each time to tensors
    a = torch.from_numpy(a).float()
    u = torch.from_numpy(u).float()
    f = torch.from_numpy(f).float()
    V = torch.from_numpy(V).float()

    height = torch.zeros((t.shape[0],1))

    # Add the bias to the total height
    height[:,0] += self.bias

    # Calculate the height for each constituent
    for i in range(self.num_constituents):
      # Add the height for the constituent to the total height
      height[:,0] += self.layers_H[i](x).squeeze() * f[i,:] * torch.cos(V[i,:] + a[i,:] * t_hours_offset + u[i,:] - self.layers_p[i](x).squeeze())
    
    return height

class TideNetSingle(nn.Module):
  """
  Astronomical tidal model for a single location. Used to fit a specific location's tidal station
  for example.
  """

  def __init__(self):
    """
    Initialize the TideNetSingle model.

    Args:
        None
    """
    super().__init__()

    # Astro data for the specific times for all constituents
    self.astro = AstroCache(verbose=False)
    self.num_constituents = self.astro.num_constituents

    # Create a simple linear model learning a bias and the amplitude and phase of each constituent
    self.bias = nn.Parameter(torch.zeros(1))
    self.layers_H = nn.Parameter(torch.randn(self.num_constituents, 1)) # Amplitude model
    self.layers_p = nn.Parameter(torch.randn(self.num_constituents, 1)) # Phase model
  
  def cache_times(self, times):
    """
    Cache the astronomical data for the specific times for all constituents.
    """
    self.astro.add_times_to_astro(times)

  def forward(self, t):
    """
    Forward pass of the TideNetSingle model.

    Args:
        t (_type_): Batime times input in seconds since epoch, UTC. Shape: [batch_size, 1]

    Returns:
        [float]: Height of the tide. Shape: [batch_size, 1]
    """

    # Height from a single constituent index n at time t is calculated as:
    #   h = H * f * cos( V + a * t + u - p )
    #  where:
    #    f, V, a, u: Astronomical data values at the specific time from self.astro.
    #    t: Hours since epoch
    #    H, p: Constituent amplitude and phase we fit with PyTorch. Varies with
    #      lat, lon and dependent on local geographic topography.
    #
    # Height from all constituents are added together to get the total height.
    # See https://github.com/sam-cox/pytides/wiki/Theory-of-the-Harmonic-Model-of-Tides

    # Load the astronomical data for the specific times for all constituents
    (a, u, f, V, t_hours_offset) = self.astro.get_astro(t.tolist())

    # Convert constants at each time to tensors
    a = torch.from_numpy(a).float()
    u = torch.from_numpy(u).float()
    f = torch.from_numpy(f).float()
    V = torch.from_numpy(V).float()

    height = torch.zeros((t.shape[0],1))

    # Add the bias to the total height
    height[:,0] += self.bias

    # Calculate the height for each constituent
    for i in range(self.num_constituents):
      # Add the height for the constituent to the total height
      height[:,0] += self.layers_H[i] * f[i,:] * torch.cos(V[i,:] + a[i,:] * t_hours_offset + u[i,:] - self.layers_p[i])
    
    return height

