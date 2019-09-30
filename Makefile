all:
	cd common/bbox; python setup_linux.py build_ext --inplace; rm -rf build; cd ../../
	cd common/chips; python setup.py build_ext --inplace; rm -rf build; cd ../../
	cd common/pycocotools; python setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd common/bbox/; rm *.so; cd ../../../
	cd common/chips/; rm *.so; cd ../../..	
	cd common/pycocotools/; rm *.so _mask.c; cd ../../
