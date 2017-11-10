all: docs

.PHONY: clean
clean:
	find . -name "*.pyc" -delete
	find . -name "*.npy" -delete

.PHONY: docs
docs:
	sphinx-apidoc -f -o docs/source/ music_classifier/ && pushd docs && make html && popd
