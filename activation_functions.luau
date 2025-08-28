--!native
--This code is written to be ran on Roblox. If you're using this code on other platforms, you may need to make modifications to make the code work.
local Activations = {}

function Activations.sigmoid(x)
	return 1 / (1 + math.exp(-x))
end

function Activations.sigmoidDerivative(output: number): number
	return output * (1 - output)
end

function Activations.tanh(x: number): number
	return math.tanh(x)
end

function Activations.tanhDerivative(output: number): number
	return 1 - output * output
end

function Activations.ReLU(x: number): number
	return (x > 0) and x or 0
end

function Activations.ReLUDerivative(x: number): number
	return (x > 0) and 1 or 0
end

function Activations.leakyReLU(x: number): number
	return (x > 0) and x or 0.01 * x
end

function Activations.leakyReLUDerivative(x: number): number
	return (x > 0) and 1 or 0.01
end

function Activations.ELU(x: number, alpha: number?): number
	alpha = alpha or 1
	return (x >= 0) and x or alpha * (math.exp(x) - 1)
end

function Activations.ELUDerivative(x: number, alpha: number?): number
	alpha = alpha or 1
	return (x >= 0) and 1 or alpha * math.exp(x)
end

local SELU_ALPHA: number = 1.6732632423543772
local SELU_LAMBDA: number = 1.0507009873554805

function Activations.SELU(x: number): number
	if x >= 0 then
		return SELU_LAMBDA * x
	else
		return SELU_LAMBDA * SELU_ALPHA * (math.exp(x) - 1)
	end
end

function Activations.SELUDerivative(x: number): number
	if x >= 0 then
		return SELU_LAMBDA
	else
		return SELU_LAMBDA * SELU_ALPHA * math.exp(x)
	end
end

function Activations.swish(x: number): number
	return x / (1 + math.exp(-x))
end

function Activations.swishDerivative(x: number): number
	local s: number = 1 / (1 + math.exp(-x))
	return s + x * s * (1 - s)
end

function Activations.hardSigmoid(x: number): number
	return math.min(1, math.max(0, 0.2 * x + 0.5))
end

function Activations.hardSigmoidDerivative(x: number): number
	return (x > -2.5 and x < 2.5) and 0.2 or 0
end

function Activations.hardSwish(x: number): number
	return x * Activations.hardSigmoid(x)
end

function Activations.hardSwishDerivative(x: number): number
	local hs: number = Activations.hardSigmoid(x)
	local dhs: number = Activations.hardSigmoidDerivative(x)
	return hs + x * dhs
end

function Activations.GELU(x: number): number
	local cdf: number = 0.5 * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x^3)))
	return x * cdf
end

function Activations.GELUDerivative(x: number): number
	local tanhTerm: number = math.tanh(0.7978845608 * (x + 0.044715 * x^3))
	local left: number = 0.5 * (1 + tanhTerm)
	local right: number = 0.0356774 * x^3 + 0.3989423 * x
	return left + x * right * (1 - tanhTerm^2)
end

function Activations.DoesFunctionExpectDerivativeInputIsSum(functionName: string): boolean
	local functionsExpectingSumInput = {
		ReLU = true,
		ReLUDerivative = true,
		ELU = true,
		ELUDerivative = true,
		SELU = true,
		SELUDerivative = true,
		leakyReLU = true,
		leakyReLUDerivative = true,
	}

	return functionsExpectingSumInput[functionName] or false
end

local functionNameMap = {}

for name, func in pairs(Activations) do
	if type(func) == "function" then
		functionNameMap[func] = name
	end
end

function Activations.GetFunctionName(func)
	return functionNameMap[func]
end

return Activations
