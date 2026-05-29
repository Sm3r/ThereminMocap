
SynthDef(\thereSynth, { |freq = 440, amp = 0.2, gate = 1, frame=0.0333|
    var sig, env;

    env = EnvGen.kr(Env.asr(0.01, 1, 0.2), gate, doneAction: 2);
    sig = SinOsc.ar(freq.lag(frame)) * amp.lag(frame) * env;

    Out.ar(0, sig!2);
}).add;


~theremin = Synth.new(\thereSynth);


OSCdef(\controller, { |msg|
	var freq, amp;

    freq = msg[1].asFloat;
    amp = msg[2].asFloat;

    [freq, amp].postln;

    ~theremin.set(\freq, freq, \amp, amp);

}, '/control');
