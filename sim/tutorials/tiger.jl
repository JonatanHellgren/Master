# S := State
# A := Action
# T := Transition
# R := Reward
# O := Observation
# Z := Observation probability dist
# γ := Discount 

using QuickPOMDPs: QuickPOMDP
using POMDPModelTools: Deterministic, Uniform, SparseCat

m = QuickPOMDP(
  states = ["left", "right"],
  actions = ["left", "right", "listen"],
  observations = ["left", "right"],
  discount = 0.95,

  transition = function(s, a)
    if a == "listen"
      return Deterministic(s) # tiger stays behind door
    else # a door is open
      return Uniform(["left", "right"]) # reset
    end
  end,

  observation = function(s, sp)
    if a == "listen"
      if sp == "left"
        return SparseCat(["left", "right"], [0.85, 0.15]) # sparse categorical
      else
        return SparseCat(["right", "left"], [0.85, 0.15]) # sparse categorical
      end
    else
      return Uniform(["left", "right"])
    end
  end,

  reward = function (s, a)
    if a == "listen"
      return -1.0
    elseif s == a # tiger was found
      return -100.0
    else # tiger was escaped
      return 10.0
    end
  end,

  initialstate = Uniform(["left", "right"])

);



