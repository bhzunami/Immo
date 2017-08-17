#!/bin/bash

set -euo pipefail
operation=$1

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )"

if [ "$operation" = "cleanup" ]; then
  echo "Cleanup the tex folder under ${dir}"
  rm -f "${dir}/Report.aux" "${dir}/Report.log" "${dir}/Report.out" "${dir}/Report.toc" "${dir}/Report.pdf" \
        "${dir}/Report.brf" "${dir}/Report.blg" "${dir}/Report.bbl"

elif [ "$operation" = "fast" ]; then
  old_dir=${PWD}
  cd "${dir}" && xelatex "${dir}/Report.tex"

elif [ "$operation" = "build" ]; then
  old_dir=${PWD}
  rm -f "${dir}/Report.aux"
  cd "${dir}" && xelatex "${dir}/Report.tex"
  bibtex Report.aux && xelatex "${dir}/Report.tex" && xelatex "${dir}/Report.tex"
  cd "${old_dir}"
  os=`uname`
  if [ "$os" = "Linux" ]; then
    : # xdg-open "${dir}/Report.pdf"
  else
    open "${dir}/Report.pdf"
  fi

elif [ "$operation" = "bib" ]; then
    cd "${dir}"
    bibtex Report.aux

else
  echo "Input parameter missing [ cleanup, build, bib]"
  exit
fi
