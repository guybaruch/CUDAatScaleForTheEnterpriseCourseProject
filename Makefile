
IDIR=./
COMPILER=nvcc

#LIBRARIES += -lcudart -lcuda -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage
LIBRARIES += -lcuda -lcudart -lnppisu -lnppif -lnppc -lculibos -lfreeimage

COMPILER_FLAGS=-I/usr/local/cuda/include -I/usr/local/cuda/lib64 \
	-I./Common -I./Common/UtilNPP ${LIBRARIES} --std c++17 \
	-g

build: bin/display_multires

bin/display_multires: src/display_multires.cu
	$(COMPILER) $(COMPILER_FLAGS) $< -o $@

clean:
	rm -f bin/*

run:
	./bin/display_multires $(ARGS)

