wget ftp://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz
wget ftp://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz.sig
gpg --verify glpk-4.65.tar.gz.sig glpk-4.65.tar.gz
gpg --recv-keys 5981E818
tar -xzvf glpk-4.65.tar.gz
cd glpk-4.65
./configure
make
