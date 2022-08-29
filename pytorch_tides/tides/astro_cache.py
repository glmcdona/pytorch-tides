from .pytides.tide import Tide, constituent
from collections import OrderedDict
import pickle
import numpy as np

class AstroCache():
  """
  Class for storing and computing astronomical data for contituent values over time.

  See https://github.com/sam-cox/pytides/wiki/Theory-of-the-Harmonic-Model-of-Tides
  for details on these constants and their use.
  """

  # Dictionary of computed values {timestamp: ([a], [u], [f], [V0], [hour_offsets])}
  astro_data = {}

  def __init__(self, constituents=None, verbose=True):
    # Use the default constituents if none are provided
    if constituents is None:
      constituents = list(OrderedDict.fromkeys(constituent.noaa))
    
    self.verbose = verbose
    if self.verbose:
      for c in constituents:
        print(f"Constituent: {c.name}")
    
    self.constituents = constituents
    self.num_constituents = len(constituents)

  def load(self, file="astro.pkl"):
    """
    Loads the astronomical data from a specified pickle file
    """
    print("Loading astronomical data from file...")

    # Load the data
    try:
      with open(file, 'rb') as f:
        self.astro_data = pickle.load(f)
        if self.verbose:
          print(f"Loaded {len(self.astro_data)} entries")
    except:
      print("Error loading astronomical data from file")
    
    return

  def save(self, file="astro.pkl"):
    """
    Save the astronomical data to a pickle file
    """
    print("Saving astronomical data to file...")

    # Save the data
    with open(file, 'wb') as f:
      pickle.dump(self.astro_data, f)
      if self.verbose:
        print(f"Saved {len(self.astro_data)} entries")
    
    return

  
  def add_times_to_astro(self, datetimes=[], chunk_length_in_days=30):
    """Compute astronomical data for a given time

    Args:
        datetimes (list, optional): List of datetimes to compute astronomical data for, UTC.
        chunk_length_in_days (int, optional): Length in days of chunks to compute astronomical data
          for. Astronomical speeds are assumed constant over this period. Defaults to 30.

    Returns:
        _type_: None
    """
    # Sort and filter to distinct times
    datetimes = sorted(list(set(datetimes)))

    # Split the times into chunks spanning less than 30 days. Slow-changing
    # constituent values are computed for each chunk to save on compute.
    t_chunks = []
    t_chunk = []
    for t_step in datetimes:
      if int(t_step.timestamp()) not in self.astro_data:
        if len(t_chunk) > 0 and (t_step - t_chunk[0]).days > chunk_length_in_days:
          t_chunks.append(t_chunk)
          t_chunk = [t_step]
        else:
          t_chunk.append(t_step)

    # Add the last chunk
    if(len(t_chunk) > 0):
      t_chunks.append(t_chunk)
      t_chunk = []
    
    if self.verbose:
      print(f"Chunks created: {len(t_chunks)}")

    # Compute the astronomical data for each chunk
    for i, t_chunk in enumerate(t_chunks):
      if self.verbose:
        print(f"Computing chunk {i} of {len(t_chunks)}...")
      t0 = t_chunk[0]
      a, u, f, V0 = Tide._prepare(self.constituents, t0, t_chunk, radians = True)

      # Notes:
      # - a: speeds of constituents. shape: [num_constituents][0]
      # - u: model phase correction fopr moons 19 year cycle. shape: [len(t_chunk)][num_constituents][0]
      # - f: model amplitude correction fopr moons 19 year cycle. shape: [len(t_chunk)][num_constituents][0]
      # - V0: Linear relation to constants. shape: [num_constituents][0]

      # Cache the results for later lookup
      for i, t in enumerate(t_chunk):
        self.astro_data[int(t.timestamp())] = (a, u[i], f[i], V0, (t-t0).total_seconds()/3600.0)

    return
  
  def get_astro(self, timestamps=[]):
    """
    Returns the astronomical data for a given time.

    Args:
        timestamps (list, optional): Time in seconds since epoch, UTC. Defaults to [].

    Returns:
        tuple: (a, u, f, V, hour_offsets) - astronomical data arrays for each time. These are used
          to compute the tide predictions at the specific timestamps.
    """
    # Returns the cached astronomical data for the given times
    a = np.zeros((self.num_constituents, len(timestamps)))
    u = np.zeros((self.num_constituents, len(timestamps)))
    f = np.zeros((self.num_constituents, len(timestamps)))
    V = np.zeros((self.num_constituents, len(timestamps)))
    hour_offsets = np.zeros((len(timestamps)))

    for i, timestamp in enumerate(timestamps):
      if int(timestamp) in self.astro_data:
        a[:, i:i+1], u[:, i:i+1], f[:, i:i+1], V[:, i:i+1], hour_offsets[i] = self.astro_data[int(timestamp)]
      else:
        print(f"ERROR: No astronomical data for {timestamp}, {int(timestamp)}")
    
    # Return numpy arrays
    return a, u, f, V, hour_offsets