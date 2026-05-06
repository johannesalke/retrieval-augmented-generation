

def test_this():
    print("Test complete!")


class InvertedIndex:
    index: dict[str,set[int]] # Maps tokens to sets of DocumentIDs
    docmap: dict[int,str]


    def __add_documents(self,doc_id: int,text:str):
        #1. Tokenize text 
        tokens = ["a"]

        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self,term):
        print("test")

    def build(self,movies):
        for m in movies:
            self.__add_documents(0,f"{m['title']} {m['description']}")
