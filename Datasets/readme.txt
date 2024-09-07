Download the lists with wget:
wget -i lnx.lst

If they are on a .xz or .gz you need to decompress them.
If they are .qcow2 images you need to decompress them using qemu-img:
	qemu-img convert -O qcow2 image.qcow2 image_uncompressed.qcow2
You might also want to .tar all files of the dataset together to simplify working with it.

Please be conscious of wasting the archives and repositories bandwidth.
Ideally download just once to do all the benchmarking and testing you need,
instead of downloading these datasets many times.

DATASETS:

- LNX (42.4GB)
	This is a dataset comprised of the source of all linux kernel versions between 3.0 and 5.19.
	No patch versions, only the release of each version, adding the patches would have made this dataset
	stupidly large and have a ridiculous duplication ratio.
	This dataset already has a REALLY high duplication ratio of ~85%.

- LNX-IMG (44.1GB)
	This one is a collection of (at the time) current/supported linux distro cloud images.
	It represent a somewhat realistic use case of a server repository hosting images with some data redundancy.
	The images are for arches amd64/x86-64, arm64/AARCH64, PPC64EL/LE, S390X and RISCV depending on distro availability.
	Different arch images of the same distro usually have significant duplicate data between themselves, and different
	distros with the same arch might have significant duplicate data in their shipped binaries.
	This dataset is again, a realistic scenario where duplicate data is not that obvious or abundant, but a good
	deduplication method should find a good amount of it.