sample_hz = 60
frame = 0

function clamp01(x)
  if x < 0 then return 0 end
  if x > 1 then return 1 end
  return x
end

function norm_ch1(v)
  local volt = v -- + 0.0000000001
  return clamp01(volt / 10.0)
end

function norm_ch2(v)
  local volt = v -- + 0.0000000001
  return clamp01((volt + 2.5) / 7.0)
end

function sample_cv(count)
  frame = frame + 1

  local v1 = input[1].volts
  local v2 = input[2].volts
  local n1 = norm_ch1(v1)
  local n2 = norm_ch2(v2)

  print(
    "cv," ..
    frame .. "," ..
    v1 .. "," ..
    v2 .. "," ..
    n1 .. "," ..
    n2
  )
end

function init()
  metro[1].event = sample_cv
  metro[1].time = 1.0 / sample_hz
  metro[1]:start()
end