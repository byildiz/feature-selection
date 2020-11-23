[paper]: https://ieeexplore.ieee.org/document/7459172
[thesis]: http://earsiv.etu.edu.tr/xmlui/handle/20.500.11851/2308

# Distinctive Interest Point Selection for Efficient Near-duplicate Image Retrieval

This repository is the official implementation of [Distinctive Interest Point Selection for Efficient Near-duplicate Image Retrieval][paper].

![Summary of the proposed method](.github/method.png 'Logo Title Text 1')

## Compiling the code

### Requirements

- Open CV
- Boost
- CMake

To compile the code run the following commands:

```bash
mkdir build
cd build && cmake ../
```

## Running the code

Please see the `run_*.sh` script files to see how to run the code.

## Results

Our proposed method has the below results on the dataset created with images at [CGFA art gallery](http://ftp.icm.edu.pl/cgfa/) using the method described in [1]:

| Method | Precision | Recall | Query Time | Indexing Time |
| ------ | --------- | ------ | ---------- | ------------- |
| [2]    | 60%       | 90%    | 59s        | 766s          |
| ours   | 95%       | 90%    | 53s        | 4325s         |

For more details please see [the paper][paper] and [my master thesis (Turkish)][thesis]

## Cite

```
@inproceedings{yildiz2016distinctive,
  title={Distinctive interest point selection for efficient near-duplicate image retrieval},
  author={Y{\i}ld{\i}z, Burak and Demirci, M Fatih},
  booktitle={2016 IEEE Southwest Symposium on Image Analysis and Interpretation (SSIAI)},
  pages={49--52},
  year={2016},
  organization={IEEE}
}
```

## References

[1] Y. Ke, R. Sukthankar, L. Huston, Y. Ke, and R. Sukthankar, “Efficient near-duplicate detection and sub-image retrieval,” in ACM Multimedia, vol. 4, no. 1, 2004, p. 5.

[2] J. J. Foo and R. Sinha, “Pruning sift for scalable near-duplicate image matching,” in Proceedings of the eighteenth conference on Australasian database-Volume 63. Australian Computer Society, Inc., 2007, pp. 63– 71.
