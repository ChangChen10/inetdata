require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('data.lua')
paths.dofile('util.lua')



