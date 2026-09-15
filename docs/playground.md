# Playground

:::DOC-TIP
This page contains a **demo simulation** of spiking neural network dynamics. It uses a small random graph with LIF (Leaky Integrate-and-Fire) neurons — not the real MaleCNS connectome. The purpose is to let you explore how spiking networks process signals, adjust parameters, and build intuition before using the real AxonWeave runtime.
:::

The playground runs entirely in your browser. No data is sent anywhere. All neuron weights, connections and parameters are generated randomly on page load.

Use the controls above to adjust the network size, connection density, LIF time constant, input signal type, frequency and amplitude. Hit **Run** to start the simulation and watch neurons spike in real time.

The code panel below the visualization shows the equivalent AxonWeave Python code that would produce the same kind of simulation on the real substrate — but note that the browser version is a simplified approximation, not the actual library execution.
