all:
	cd utils/pycocotools; python setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd utils/pycocotools/; rm *.so _mask.c; cd ../../
	