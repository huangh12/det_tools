all:
	cd common/bbox; python setup_linux.py build_ext --inplace; rm -rf build; cd ../../
	cd isn_tool/utils/chips; python setup.py build_ext --inplace; rm -rf build; cd ../../../
	cd eval_tool/utils/pycocotools; python setup.py build_ext --inplace; rm -rf build; cd ../../../
clean:
	cd common/bbox/; rm *.so; cd ../../
	cd isn_tool/utils/chips; rm *.so; cd ../../..
	cd eval_tool/utils/pycocotools; rm *.so _mask.c; cd ../../../
