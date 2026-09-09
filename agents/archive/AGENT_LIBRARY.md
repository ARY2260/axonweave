# Library Agent Playbook

## API rules

Use typed public interfaces, stable error codes and explicit configuration. Avoid hidden global state.

## Data rules

Never add raw upstream MaleCNS data to source control. Substrate artifacts are provisioned and cached separately.

## Backend rules

Adapters must use the host framework's tensor, sparse and device APIs. Do not implement a second tensor runtime inside AxonWeave.

## Scientific rules

A source observation, a fitted parameter and a modeling assumption must have separate representations.

## Testing rules

Test graph shape/index invariants, serialization, error behavior, backend numerical behavior, and device capability reporting independently.
