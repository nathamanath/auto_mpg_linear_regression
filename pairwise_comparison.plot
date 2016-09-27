set term png size 1920,1080
set output 'pairwise_comparison.png'

# Read data in from files
N=`awk 'NR==1 {print NF}' auto_mpg_expanded.data`
labels="`cat auto_mpg_expanded.names`"

# Work out how many plots will be produced
count=0
do for [i=2:N] {
  do for [j=i+1:N] {
    count = count +1
  }
}

# will present in rxr grid
r = ceil(sqrt(count))

set multiplot layout r,r

# Plot each column against all other columns one time
do for [i=2:N] {

  set xlabel word(labels, i)

  do for [j=i+1:N] {

    set ylabel word(labels, j)

    plot "auto_mpg_expanded.data" u i:j title ""
  }
}
