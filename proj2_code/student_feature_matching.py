import numpy as np


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # pytest unit_tests/ -k test_compute_dists
    # pytest unit_tests/ -k test_feature_matching_speed


    """

    # fast approach but uses too much memory ( > 8G)

    feat_dim = features1.shape[1]
    n = features1.shape[0]
    m = features2.shape[0]
    tiled_1 = np.tile(features1,m).reshape(n, m, feat_dim)
    tiled_2 = np.tile(features2.flatten(), n).reshape(n, m, feat_dim)

    diff = (tiled_1 - tiled_2).reshape(m * n,feat_dim)
    dists = np.linalg.norm(diff, axis=1).reshape(n,m)
    #dists = np.sqrt(np.sum(np.power((tiled_1 - tiled_2),2), axis = 2)).reshape(n,m)
  
    """

    # slow approach that uses ~50M memory

    feat_dim = features1.shape[1]
    n = features1.shape[0]
    m = features2.shape[0]

    dists = np.zeros((n,m))

    for i in range(n):
      for j in range(m):
        dists[i][j] = np.linalg.norm(features1[i] - features2[j])
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # dists: numpy array of shape (n,m)
    dists = compute_feature_distances(features1, features2)

    # compute ratios
    smallest_two_indices = np.argsort(dists, axis = 0)[0,:]
    samllest_two_c = np.sort(dists, axis = 0)[:2,:]
    all_confidences = samllest_two_c[0] / samllest_two_c[1]

    # threshold 0.8 in original paper, but tewak it to get 80% on Notre Dame image
    matches, confidences = [],[]
    threshold = 0.8

    for index in range(len(all_confidences)):
      if all_confidences[index] < threshold:
        matches.append([smallest_two_indices[index],index])
        confidences.append(all_confidences[index])
    
    matches = np.asarray(matches)
    confidences = np.asarray(confidences)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
