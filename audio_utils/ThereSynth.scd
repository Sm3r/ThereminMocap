
s.options.outDevice = "Windows DirectSound : DELL S2421H (NVIDIA High Definition Audio)";

s.waitForBoot {
    SynthDef(\thereSynth, { |freq = 440, amp = 0.2, gate = 1, frame=0.0333|
        var sig, env;

        env = EnvGen.kr(Env.asr(0.01, 1, 0.2), gate, doneAction: 2);
        sig = SinOsc.ar(freq.lag(frame)) * amp.lag(frame) * env;

        Out.ar(0, sig!2);
    }).add;

    s.sync;

    ~theremin = Synth.new(\thereSynth);

    OSCdef(\pitchCtrl, { |msg|
        var freq = msg[1].asFloat;
        ~theremin.set(\freq, freq);
    }, '/pitch');

    OSCdef(\volumeCtrl, { |msg|
        var amp = msg[1].asFloat;
        ~theremin.set(\amp, amp);
    }, '/volume');
};
