#!/bin/bash

# Set up environment variables
export JAX_PLATFORM_NAME=cpu  # Use CPU for testing to avoid GPU-specific issues

# Install test dependencies if they don't exist
pip install pytest pytest-cov pytest-xdist pytest-randomly colorama

# Create directory for test reports
mkdir -p test_reports

# Run tests with coverage
python -m pytest tests/ \
    --cov=pcx \
    --cov-report=term \
    --cov-report=html:test_reports/coverage \
    --cov-report=xml:test_reports/coverage.xml \
    -v $@

# Exit with the pytest return code
exit $? 