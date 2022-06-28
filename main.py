import typing
import json
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from sentence_transformers import (
    SentenceTransformer,
    util
)
from torch import Tensor


def encode_images(model: SentenceTransformer) -> typing.Tuple[typing.List[str], Tensor]:
    """Given a model, encode some images stored locally.
    """
    image_paths = list(glob.glob("./images/dog_api/**/*.jpg"))
    images = [
        Image.open(file)
        for file in image_paths
    ]

    return image_paths, model.encode(images, batch_size=128, show_progress_bar=True, convert_to_tensor=True)


def cluster(embeddings: Tensor, threshold: float = 0.9, min_community_size: int = 2, init_max_size: int = 500):
    """
    Given tensor embeddings of the images, cluster them into communities using a sort-of greedy method.
    https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/image-search/Image_Clustering.ipynb


    """
    # Compute cosine similarity scores
    cos_scores = util.cos_sim(embeddings, embeddings)

    # Minimum size for a community
    top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

    # Filter for rows >= min_threshold
    extracted_communities = []

    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            # Only check top k most similar entries
            top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)
            else:
                # Iterate over all entries (slow)
                for idx, val in enumerate(cos_scores[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)

            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)

    return unique_communities


def make_collage(filenames: typing.List[str], out_file: str):
    """Make an image grid collage of given paths to images.
    """
    images = []

    for filename in filenames:
        images.append(Image.open(filename))

    fig = plt.figure(figsize=(40., 40.))

    grid = ImageGrid(
        fig, 
        111,
        nrows_ncols=(10, 8),
        axes_pad=0.1,
    )

    for ax, im in zip(grid, images):
        ax.imshow(im)
    
    plt.savefig(out_file)
    plt.close()
    return


def main():
    """
    Image clustering example using sentence transformer.

    Essentially, we cluster images based on their embeddings
    in the vector space of the transformer model.

    """

    # Load model
    model = SentenceTransformer('clip-ViT-L-14')

    # Compute tensor embeddings. This can be pre-computed and stored on disk.
    image_paths, image_embeddings = encode_images(model)

    # Cluster the tensor embeddings of the corresponding images.

    # Gotta ensure init_max_size is smaller than the number of images. Higher the threshold, the more reliable
    # the cluster (with the risk of over-clustering).
    comms = cluster(image_embeddings, threshold=0.95, min_community_size=10, init_max_size=500)

    # Collect the corresponding images of the communities found.
    result = []
    for community_idx, community in enumerate(sorted(comms, key=lambda x: len(x), reverse=True)):
        current_cluster = []
        for member in community:
            current_cluster.append(image_paths[member])

        result.append(current_cluster)

        print(f"Finished clustering cluster: {community_idx:02d}")
        make_collage(current_cluster, f"results/cluster_{community_idx:02d}.jpg")
        print(f"Finished collage for cluster: {community_idx:02d}")

        # Let's just keep a few communities. Not a lot.
        if community_idx >= 20:
            break

    with open("results/clusters.json", "w") as f:
        json.dump(result, f, indent=4)



if __name__ == '__main__':
    main()
