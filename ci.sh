#!/bin/sh

set -e

black --diff --check $(git ls-files '*.py')
mypy --strict $(git ls-files '*.py')
pylint $(git ls-files '*.py')
