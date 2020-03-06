cd $(dirname "$0") || exit
cat ./official-2014.0.m2 | grep "^S" | cut -d' '  -f2- > ./official-2014.0.m2.src
cat ./official-2014.1.m2 | grep "^S" | cut -d' '  -f2- > ./official-2014.1.m2.src
cat ./official-2014.combined.m2 | grep "^S" | cut -d' '  -f2- > ./official-2014.combined.m2.src

