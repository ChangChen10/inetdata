require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'fbcunn'
require 'nn'
require 'cudnn'
vstruct = require "vstruct"

torch.setdefaulttensortype('torch.FloatTensor')
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
opt.dataType = 'val'
paths.dofile('data.lua')
paths.dofile('util.lua')
convModule = torch.load('convModule.t7')

local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.LongTensor()
inputs = torch.CudaTensor()
labels = torch.CudaTensor()

function getFeatures(inputsThread, labelsThread, numBatch)
  receiveTensor(inputsThread, inputsCPU)
  receiveTensor(labelsThread, labelsCPU)
  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)
  
  features = convModule:forward(inputs)
  -- TODO: using these features to train
  dim = features:size()
  dim1 = dim[1]
  dim2 = dim[2]*dim[3]*dim[4]
  features:resize(dim1,dim2)
  local filename = "val/heldout." .. numBatch .. ".fea"
  file = io.open (filename , "w+")
  --print(dim1)
  vstruct.write(">u4", file, {dim1})
  for i = 1, dim1 do
    vstruct.write(">u4", file, {dim2})
    for j = 1, dim2 do
      vstruct.write(">f4", file, {features[i][j]})
    end
  end
  file:close() 
  filename = "val/heldout." .. numBatch .. ".lab"
  file = io.open (filename , "w+")
  vstruct.write(">u4", file, {dim1})
  for i = 1, dim1 do
    vstruct.write(">u4", file, {labels[i]})
  end
  file:close();
end

local n = nTest -- nTest is set in 1_data.lua
local k = math.floor(n/opt.testBatchSize)
for i=opt.from, opt.to do
  print('batch ' .. i .. ' of ' .. k+1)
  local indexStart = (i-1) * opt.testBatchSize + 1
  local indexEnd = (indexStart + opt.testBatchSize - 1)
  if i == k + 1 then
    indexEnd = n
  end

  donkeys:addjob(
     -- work to be done by donkey thread
     function()
        local inputs, labels = testLoader:get(indexStart, indexEnd)
        return sendTensor(inputs), sendTensor(labels), i
     end,
     -- callback that is run in the main thread once the work is done
     getFeatures
  )
  if i % 5 == 0 then
     donkeys:synchronize()
     collectgarbage()
  end
end
