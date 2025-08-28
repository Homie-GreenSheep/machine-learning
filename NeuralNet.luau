--!native
--This code is written to be ran on Roblox. If you're using this code on other platforms, you may need to make modifications to make the code work.
--[[
NeuralNet Module Documentation

A lightweight neural network library in Lua supporting customizable neurons, layers,
and training via backpropagation.

=======================================
Types
=======================================

Neuron
  - Weights: array<number>               -- weights for each input
  - Bias: number                         -- bias term (default: random between -1 and 1)
  - ActivationFunction: function         -- function(number) -> number (default: LeakyReLU)
  - DerivativeFunction: function         -- derivative of activation function (default: LeakyReLU derivative)
  - DerivativeInputIsSum: boolean?       -- whether derivative expects the weighted sum as input
  - ActivationName: string               -- name of the activation function
  - DerivativeName: string               -- name of the derivative function

Layer
  - Neurons: array<Neuron>               -- neurons in the layer

NeuralNetwork
  - LearningRate: number                 -- learning rate for training
  - Network: array<Layer>                -- layers of the network

=======================================
Functions
=======================================

nn.new(learningRate: number) -> NeuralNetwork
  Creates a new neural network with a given learning rate.

  Parameters:
    learningRate (number) - controls speed of training updates
  Returns:
    NeuralNetwork - new empty network

nn.CreateNeuron(weights: array<number>, bias?: number, activation?: function, derivative?: function, activationName?: string) -> Neuron
  Creates a neuron with weights, optional bias, activation, and derivative functions.

  Parameters:
    weights (array<number>)           - array of weights
    bias (number, optional)           - bias term (default: random between -1 and 1)
    activation (function, optional)   - activation function (default: LeakyReLU)
    derivative (function, optional)   - derivative function (default: LeakyReLU derivative)
    activationName (string, optional) - name of the activation function
  Returns:
    Neuron

nn.CreateLayer(neurons: array<Neuron>) -> Layer
  Creates a layer from a list of neurons.

  Parameters:
    neurons (array<Neuron>) - list of neurons (must contain at least one)
  Returns:
    Layer

nn.AddLayer(network: NeuralNetwork, layer: Layer) -> NeuralNetwork
  Adds a layer to the neural network.

  Parameters:
    network (NeuralNetwork) - the network to modify
    layer (Layer)           - layer to add (must contain at least one neuron)
  Returns:
    NeuralNetwork - updated network

nn.FeedForward(network: NeuralNetwork, inputs: array<number>) -> array<number>
  Feeds inputs through the network and returns outputs.

  Parameters:
    network (NeuralNetwork) - the network instance
    inputs (array<number>)   - input values
  Returns:
    array<number> - output values

nn.Train(network: NeuralNetwork, inputs: array<number>, expected: array<number>) -> NeuralNetwork
  Performs one training step using backpropagation.

  Parameters:
    network (NeuralNetwork) - the network instance
    inputs (array<number>)   - input values
    expected (array<number>) - expected output values
  Returns:
    NeuralNetwork - updated network

nn.SaveModel(network: NeuralNetwork) -> string
  Serializes the network into a JSON string.

  Parameters:
    network (NeuralNetwork) - the network instance
  Returns:
    string - JSON-encoded network

nn.LoadModel(jsonString: string) -> NeuralNetwork
  Loads a network from a JSON string previously created by nn.SaveModel.

  Parameters:
    jsonString (string) - JSON-encoded network
  Returns:
    NeuralNetwork - restored network

=======================================
Example Usage
=======================================

local nn = require("nn")

local net = nn.new(0.01)

local neuron1 = nn.CreateNeuron({0.5, -0.3})
local neuron2 = nn.CreateNeuron({-0.8, 0.2})
local layer1 = nn.CreateLayer({neuron1, neuron2})

net = nn.AddLayer(net, layer1)

local output = nn.FeedForward(net, {1, 0})
print(output[1], output[2])

net = nn.Train(net, {1, 0}, {1, 0})

local jsonModel = nn.SaveModel(net)
local restoredNet = nn.LoadModel(jsonModel)
]]

local activationFunctions = require(script.ActivationFunctions)
local nn = {}

export type Neuron = {
	Weights: {number},
	Bias: number,
	ActivationFunction: (number) -> number,
	DerivativeFunction: (number) -> number,
	DerivativeInputIsSum: boolean?,
	ActivationName: string,
	DerivativeName: string,
}

export type Layer = {
	Neurons: {Neuron},
}

export type NeuralNetwork = {
	LearningRate: number,
	Network: {Layer},
}

function nn.new(learningRate: number): NeuralNetwork
	return {
		LearningRate = learningRate,
		Network = {},
	}
end

function nn.CreateNeuron(
	weights: {number},
	bias: number?,
	activation: ((number) -> number)?,
	derivative: ((number) -> number)?,
	activationName: string?
): Neuron
	local function randomBias()
		return (math.random() - 0.5) * 2
	end

	activation = activation or activationFunctions.leakyReLU
	derivative = derivative or activationFunctions.leakyReLUDerivative

	local functionName = activationName or activationFunctions.GetFunctionName(activation)
	local derivativeName = functionName and (functionName .. "Derivative") or nil
	local derivativeInputIsSum = functionName and activationFunctions.DoesFunctionExpectDerivativeInputIsSum(functionName) or false

	return {
		Weights = weights,
		Bias = (bias ~= nil) and bias or randomBias(),
		ActivationFunction = activation,
		DerivativeFunction = derivative,
		DerivativeInputIsSum = derivativeInputIsSum,
		ActivationName = functionName,
		DerivativeName = derivativeName,
	}
end

function nn.CreateLayer(neurons: {Neuron}): Layer
	assert(#neurons > 0, "Layer must contain at least one neuron")
	return {Neurons = neurons}
end

function nn.AddLayer(network: NeuralNetwork, layer: Layer): NeuralNetwork
	assert(#layer.Neurons > 0, "Layer must have at least one neuron")
	table.insert(network.Network, layer)
	return network
end

function nn.FeedForward(network: NeuralNetwork, inputs: {number}): {number}
	local currentInputs = inputs
	for _, layer in ipairs(network.Network) do
		local outputs = {}
		for _, neuron in ipairs(layer.Neurons) do
			local sum = neuron.Bias
			for i, weight in ipairs(neuron.Weights) do
				sum += currentInputs[i] * weight
			end
			table.insert(outputs, neuron.ActivationFunction(sum))
		end
		currentInputs = outputs
	end
	return currentInputs
end

function nn.Train(network: NeuralNetwork, inputs: {number}, expected: {number}): NeuralNetwork
	local activations = {inputs}
	local sums = {}

	for _, layer in ipairs(network.Network) do
		local layerSums = {}
		local layerOutputs = {}
		local prevActivation = activations[#activations]
		for _, neuron in ipairs(layer.Neurons) do
			local sum = neuron.Bias
			for i, weight in ipairs(neuron.Weights) do
				sum += prevActivation[i] * weight
			end
			table.insert(layerSums, sum)
			table.insert(layerOutputs, neuron.ActivationFunction(sum))
		end
		table.insert(sums, layerSums)
		table.insert(activations, layerOutputs)
	end

	local deltas = {}
	local lastLayerIndex = #network.Network

	local outputActivations = activations[#activations]
	local outputLayer = network.Network[lastLayerIndex]
	local outputDeltas = {}
	for i, neuron in ipairs(outputLayer.Neurons) do
		local error = expected[i] - outputActivations[i]
		local derivativeInput = neuron.DerivativeInputIsSum and sums[lastLayerIndex][i] or outputActivations[i]
		local delta = error * neuron.DerivativeFunction(derivativeInput)
		outputDeltas[i] = delta
	end
	deltas[lastLayerIndex] = outputDeltas

	for layerIndex = lastLayerIndex - 1, 1, -1 do
		local layer = network.Network[layerIndex]
		local nextLayer = network.Network[layerIndex + 1]
		local layerDeltas = {}
		for i, neuron in ipairs(layer.Neurons) do
			local sum = 0
			for j, nextNeuron in ipairs(nextLayer.Neurons) do
				sum += nextNeuron.Weights[i] * deltas[layerIndex + 1][j]
			end
			local derivativeInput = neuron.DerivativeInputIsSum and sums[layerIndex][i] or activations[layerIndex + 1][i]
			local delta = sum * neuron.DerivativeFunction(derivativeInput)
			layerDeltas[i] = delta
		end
		deltas[layerIndex] = layerDeltas
	end

	for layerIndex, layer in ipairs(network.Network) do
		local layerDeltas = deltas[layerIndex]
		local inputActivation = activations[layerIndex]
		for i, neuron in ipairs(layer.Neurons) do
			for w = 1, #neuron.Weights do
				neuron.Weights[w] += network.LearningRate * layerDeltas[i] * inputActivation[w]
			end
			neuron.Bias += network.LearningRate * layerDeltas[i]
		end
	end

	return network
end

local HttpService = game:GetService("HttpService")

function nn.SaveModel(network: NeuralNetwork): string
	local copy = {
		LearningRate = network.LearningRate,
		Network = {}
	}

	for _, layer in ipairs(network.Network) do
		local layerCopy = {Neurons = {}}
		for _, neuron in ipairs(layer.Neurons) do
			local actName = activationFunctions.GetFunctionName(neuron.ActivationFunction) or "leakyReLU"
			local derName = activationFunctions.GetFunctionName(neuron.DerivativeFunction) or "leakyReLUDerivative"
			table.insert(layerCopy.Neurons, {
				Weights = neuron.Weights,
				Bias = neuron.Bias,
				ActivationFunction = actName,
				DerivativeFunction = derName,
				DerivativeInputIsSum = neuron.DerivativeInputIsSum,
			})
		end
		table.insert(copy.Network, layerCopy)
	end

	return HttpService:JSONEncode(copy)
end

function nn.LoadModel(jsonString: string): NeuralNetwork
	local data = HttpService:JSONDecode(jsonString)
	for _, layer in ipairs(data.Network) do
		for _, neuron in ipairs(layer.Neurons) do
			local actName = neuron.ActivationFunction
			local derName = neuron.DerivativeFunction
			if type(actName) == "string" then
				neuron.ActivationFunction = activationFunctions[actName]
				neuron.DerivativeFunction = activationFunctions[derName]
				neuron.DerivativeInputIsSum = activationFunctions.DoesFunctionExpectDerivativeInputIsSum(actName)
			end
		end
	end
	return data
end

return nn
