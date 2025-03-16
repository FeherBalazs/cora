# PCX Test Suite

This directory contains the test suite for the PCX library, which verifies the functionality and correctness of the various components of the library.

## Test Organization

The tests are organized as follows:

- `tests/pcx/`: Tests for the core functionality of PCX
  - `test_core.py`: Tests for core classes and utilities
  - `test_nn.py`: Tests for neural network layers
  - `test_optim.py`: Tests for optimization classes
  - `test_vode.py`: Tests for Vode (Vectorised Node) implementation
  - `test_energy_module.py`: Tests for the EnergyModule class
  - `test_integration.py`: Integration tests combining multiple components

## Running the Tests

### Running All Tests

To run all tests with coverage reporting, use the provided shell script:

```bash
./run_tests.sh
```

This will:
1. Run all tests in the `tests/` directory
2. Generate a coverage report in the terminal
3. Generate HTML and XML coverage reports in the `test_reports/` directory

### Running Specific Tests

To run a specific test file:

```bash
python -m pytest tests/pcx/test_nn.py -v
```

To run a specific test function:

```bash
python -m pytest tests/pcx/test_nn.py::test_linear_forward -v
```

## Coverage

The test suite aims to maintain high code coverage to ensure robustness of the PCX library. Coverage reports are automatically generated when running the tests using the provided shell script.

To view the HTML coverage report:

```bash
open test_reports/coverage/index.html
```

## Contributing New Tests

When adding new functionality to PCX, please also add corresponding tests. Follow these guidelines:

1. Place tests in the appropriate file based on the component being tested
2. Use descriptive test names that clearly indicate what's being tested
3. Include docstrings that explain the purpose of the test
4. Ensure tests are deterministic (e.g., use fixed random seeds)
5. Test both normal operation and edge cases

### Test Structure

Each test should follow this general structure:

```python
def test_component_functionality():
    """Test that [component] correctly handles [functionality]."""
    # Setup
    component = Component(parameters)
    
    # Exercise
    result = component.function()
    
    # Verify
    assert result == expected_result
```

## Debugging Failed Tests

When a test fails, the error message will include:
- The location of the failing test
- The assertion that failed
- The expected and actual values

For more detailed debugging, you can use the `-v` (verbose) flag:

```bash
python -m pytest tests/pcx/test_file.py -v
```

## Prerequisites

The test suite requires the following packages:
- pytest
- pytest-cov (for coverage reporting)
- pytest-xdist (for parallel test execution)
- pytest-randomly (for randomized test order)
- colorama (for colored output)

These dependencies are automatically installed when running `./run_tests.sh`. 