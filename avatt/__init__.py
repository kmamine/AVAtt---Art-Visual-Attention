"""AVAtt: free-viewing eye-tracking data on paintings.

    >>> import avatt
    >>> ds = avatt.load()                      # downloads the dataset from the Hugging Face Hub once
    >>> ds.metadata.head()
    >>> fix = ds.scanpaths("avatt_0001")       # list of per-observer DataFrames
    >>> sal = ds.saliency_map("avatt_0001")
    >>> avatt.metrics.nss(model_map, ds.fixation_map("avatt_0001"))
"""
from . import geometry, maps, metrics
from .data import AVAtt, download, load

__version__ = "2.0.0"
__all__ = ["AVAtt", "__version__", "download", "geometry", "load", "maps", "metrics"]
