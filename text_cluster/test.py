# get clusters using affinity propagation
    ap_obj, clusters = affinity_propagation(feature_matrix=feature_matrix)
    book_data['Cluster'] = clusters

    # get the total number of books per cluster
    c = Counter(clusters)
    print(c.items())

    # get total clusters
    total_clusters = len(c)
    print('Total Clusters:', total_clusters)

    cluster_data = get_cluster_data(clustering_obj=ap_obj,
                                    book_data=book_data,
                                    feature_names=feature_names,
                                    num_clusters=total_clusters,
                                    topn_features=5)

    print_cluster_data(cluster_data)

    plot_clusters(num_clusters=num_clusters,
                  feature_matrix=feature_matrix,
                  cluster_data=cluster_data,
                  book_data=book_data,
                  plot_size=(16, 8))


    # build ward's linkage matrix
    linkage_matrix = ward_hierarchical_clustering(feature_matrix)
    # plot the dendrogram
    plot_hierarchical_clusters(linkage_matrix=linkage_matrix,
                           book_data=book_data,
                           figure_size=(8, 10))
    print('......')