import argparse
import hybrid_search as hs
import lib.multimodal_search as mms




def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify that image embeddings can be performed"
    )
    verify_image_embedding_parser.add_argument("img_path", type=str, help="Path for the image to be embedded")

    image_search_parser = subparsers.add_parser("image_search", help="Summarize search results")
    image_search_parser.add_argument("img_path", type=str, help="Path to the image")
    



    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            #MMS = mms.MultimodalSearch()
            mms.verify_image_embedding(args.img_path)
            #= MMS.embed_image(args.img_path)
        case "image_search":
            results = mms.image_search_command(args.img_path)
            for i,res in enumerate(results):
                print(f"{i+1}. {res["title"]} (similarity: {res["similarity_score"]:.3f})")
                print("   "+res["description"[:100]])


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()