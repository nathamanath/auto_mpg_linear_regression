#!/bin/bash

# Pre-process auto_mpg.data ready for learning

DATA_FILE=./auto_mpg.data

# Normalize spacing
# Normalize brand names - fix typos, no abbreviations
# Extract brand name from car name
# Replace brand name with id
auto_mpg_cleaned=$(cat $DATA_FILE \
| sed -e 's/\(\s\{2,\}\|\t\+\)/\ \ \ \ /g' \
| sed -f brands.sed \
| awk -F'    ' 'match($9, /[A-Za-z\-]+/) {print $0 "    \"" substr($9, RSTART, RLENGTH)"\""}' \
| sed -f brands_ids.sed)

# Put car names in file with ids
echo "$auto_mpg_cleaned" | awk -F '    ' '{print $9}' > cars.names

# Prepend car id column to data
auto_mpg_cleaned=$(echo "$auto_mpg_cleaned" | awk -F'    ' '{
  $9=""
  print NR" "$0
}')

# Write expanded dataset to file
# Used for plotting feature comparisons
echo "$auto_mpg_cleaned" > auto_mpg_expanded.data

# Strip any unwanted columns
# Shuffle rows
auto_mpg_cleaned=$(echo "$auto_mpg_cleaned" | awk '{

  # Make binary column per brand
  brands=""

  for(i=1;i<37; i++) {
    if(i == $9) {
      a=1
    } else {
      a=0
    }

    brands=brands"    "a
  }

  # Remove unwanted columns
  $10="" # brand id
  $9="" # region id
  $7="" # acceleration
  $5="" # horse power

  print $0 brands

}' \
| sed s/?/0/ \
| sort -R)

# Split into train, test, and validation sets (60:20:20)
m=$(wc -l $DATA_FILE | awk '{print $1}')

test_lines=$(echo "($m / 100) * 20" | bc)
cv_lines=$test_lines
train_lines=$(echo "$m - $cv_lines - $test_lines" | bc)

echo "$auto_mpg_cleaned" | head -n$train_lines > linear_regression/data/train.data
echo "$auto_mpg_cleaned" | tail -n$cv_lines > linear_regression/data/cv.data
echo "$auto_mpg_cleaned" | tail -n$(echo "$cv_lines + $test_lines" | bc) | head -n$test_lines > linear_regression/data/test.data
