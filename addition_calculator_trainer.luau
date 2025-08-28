--!native
--This code is written to be ran on Roblox. If you're using this code on other platforms, you may need to make modifications to make the code work.
local RunService = game:GetService("RunService")
local rs = game:GetService("ReplicatedStorage")

local nn = require(rs:WaitForChild("NeuralNet"))
local activations = require(rs:WaitForChild("NeuralNet").ActivationFunctions)

local function randomWeight()
	return (math.random() - 0.5) * 2
end

local function shuffle(t)
	for i = #t, 2, -1 do
		local j = math.random(i)
		t[i], t[j] = t[j], t[i]
	end
end

local function oneHot(index, size)
	local vec = {}
	for i = 1, size do
		vec[i] = (i == index) and 1 or 0
	end
	return vec
end

local learningRate = 0.005
local maxEpochs = 75000
local errorThreshold = 1e-6
local epochsPerStep = 50
local tolerance = 0.25

local inputSize = 20
local outputSize = 1 --single scalar output
local layerSizes = {inputSize, 32, outputSize}

local network = nn.new(learningRate)

for layerIndex = 2, #layerSizes do
	local inputCount = layerSizes[layerIndex - 1]
	local outputCount = layerSizes[layerIndex]
	local neurons = {}

	for _ = 1, outputCount do
		local weights = {}
		for _ = 1, inputCount do
			table.insert(weights, randomWeight())
		end
		--Output layer: linear activation (no activation)
		local activation = (layerIndex == #layerSizes) and function(x) return x end or activations.swish
		local activationDerivative = (layerIndex == #layerSizes) and function(x) return 1 end or activations.swishDerivative
		table.insert(neurons, nn.CreateNeuron(weights, nil, activation, activationDerivative))
	end

	nn.AddLayer(network, nn.CreateLayer(neurons))
end

local trainingData = {}
local maxSum = 18
for a = 0, 9 do
	for b = 0, 9 do
		local input = {}
		for _, v in ipairs(oneHot(a + 1, 10)) do table.insert(input, v) end
		for _, v in ipairs(oneHot(b + 1, 10)) do table.insert(input, v) end
		local expected = {(a + b) / maxSum} --normalized output
		table.insert(trainingData, {inputs = input, expected = expected})
	end
end

local currentEpoch = 0
local done = false
local lastError = math.huge
local heartbeatConnection

local function computeTotalError()
	local errSum = 0
	for _, data in ipairs(trainingData) do
		local out = nn.FeedForward(network, data.inputs)
		local err = data.expected[1] - out[1]
		errSum += err * err
	end
	return errSum / #trainingData
end

local function printFinalResults()
	print("=== Calculator Test ===")
	local totalWrong = 0
	for a = 0, 9 do
		for b = 0, 9 do
			local input = {}
			for _, v in ipairs(oneHot(a + 1, 10)) do table.insert(input, v) end
			for _, v in ipairs(oneHot(b + 1, 10)) do table.insert(input, v) end
			local output = nn.FeedForward(network, input)
			local sumPred = output[1] * maxSum
			local expected = a + b

			local correct = math.abs(sumPred - expected) <= tolerance
			if not correct then
				totalWrong += 1
				break
			end
			print(string.format(
				"%d + %d = %.4f (Expected: %d) %s",
				a, b, sumPred, expected,
				correct and "✅" or "❌"))
		end
		if totalWrong >= 1 then
			break
		end
	end
	if totalWrong >= 1 then
		local clone = script:Clone()
		clone.Parent = script.Parent
		script:Destroy()
	else
		print(nn.SaveModel(network))
	end
end


print(string.format("Epoch %d | Error: %.17f", 0, computeTotalError()))

--If you try to do it without the heartbeat stuff, Roblox will time out the script and it may freeze, and it won't finish.

heartbeatConnection = RunService.Heartbeat:Connect(function()
	if done then
		heartbeatConnection:Disconnect()
		return
	end

	if currentEpoch >= maxEpochs then
		done = true
		print("Reached max epochs, stopping training.")
		printFinalResults()
		return
	end

	for i = 1, epochsPerStep do
		if currentEpoch >= maxEpochs then break end

		shuffle(trainingData)
		for _, data in ipairs(trainingData) do
			nn.Train(network, data.inputs, data.expected)
		end

		currentEpoch += 1

		if currentEpoch % 250 == 0 then
			local err = computeTotalError()
			--print(string.format("Epoch %d | Error: %.17f", currentEpoch, err))

			if err <= errorThreshold then
				print("🎉 Training complete!")
				done = true
				printFinalResults()
				break
			elseif err > lastError then
				print(string.format("Error increased from %.17f to %.17f, stopping training early.", lastError, err))
				done = true
				printFinalResults()
				break
			end

			lastError = err
		end
	end
end)
