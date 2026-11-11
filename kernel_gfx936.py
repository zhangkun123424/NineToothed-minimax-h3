import ninetoothed as nt
import ninetoothed.language as ntl
def arrangement(timestep, max_period, cos_output, sin_output, embedding_dim):
    half = embedding_dim // 2
    timestep = timestep.unsqueeze(1).tile((1, 1))
    timestep.dtype = timestep.dtype.squeeze((0, 1))
    cos_output = cos_output.tile((1, half))
    cos_output.dtype = cos_output.dtype.squeeze(0)
    sin_output = sin_output.tile((1, half))
    sin_output.dtype = sin_output.dtype.squeeze(0)
    return (timestep, max_period, cos_output, sin_output)

def application(timestep, max_period, cos_output, sin_output):
    channel = ntl.arange(0, cos_output.shape[0])
    time_fp32 = ntl.cast(timestep, ntl.float32)
    period_fp32 = ntl.cast(max_period, ntl.float32)
    index_fp32 = ntl.cast(channel, ntl.float32)
    exponent = -ntl.cast(ntl.extra.libdevice.log(ntl.cast(period_fp32, ntl.float64)), ntl.float32) * index_fp32 * (1.0 / cos_output.shape[0])
    angle = time_fp32 * ntl.extra.libdevice.exp(exponent)
    cos_output = ntl.cos(angle)
    sin_output = ntl.extra.libdevice.sin(angle)