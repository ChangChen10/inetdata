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
convModule = torch.load('convModule.t7')

local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.LongTensor()
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function getFeatures(inputsThread, labelsThread)
   receiveTensor(inputsThread, inputsCPU)
   receiveTensor(labelsThread, labelsCPU)
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)


   features = convModule:forward(inputs)
   -- TODO: using these features to train

end

for i=1,nTest/nTest do -- nTest is set in 1_data.lua
  local indexStart = (i-1) * opt.testBatchSize + 1
  local indexEnd = (indexStart + opt.testBatchSize - 1)
  donkeys:addjob(
     -- work to be done by donkey thread
     function()
        local inputs, labels = testLoader:get(indexStart, indexEnd)
        return sendTensor(inputs), sendTensor(labels)
     end,
     -- callback that is run in the main thread once the work is done
     getFeatures
  )
  if i % 5 == 0 then
     donkeys:synchronize()
     collectgarbage()
  end

end


