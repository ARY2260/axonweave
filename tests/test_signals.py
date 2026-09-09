from axonweave.signals import SignalPolicy, NeurotransmitterGain, LeakyPropagation

def test_explicit_signal_policy():
    p=SignalPolicy({'a':1.5}, default_gain=-0.25)
    assert p.gain('a') == 1.5
    assert p.gain('unknown') == -0.25
    assert NeurotransmitterGain(p).apply('a',2)==3

def test_leak():
    assert LeakyPropagation(.5).step(2,3)==4
