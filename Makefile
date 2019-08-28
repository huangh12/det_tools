all:
	cd utils/bbox; python setup_linux.py build_ext --inplace; rm -rf build; cd ../../
	cd utils/chips; python setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd utils/bbox/; rm *.so; cd ../../../
	cd utils/chips/; rm *.so; cd ../../..	
