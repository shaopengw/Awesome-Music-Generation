import numpy as np
import faiss
import time

class FaissDatasetBuilder:
    def __init__(self, data, index_type=None, nlist=None, nprobe=None, batch_size=None):
        self.data = data
        self.dimension = self.data.shape[1]
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = None
        self.batch_size = batch_size

    def build_index(self):
        print("WangHaoyu: Start to build index")
        if self.index_type == 'Flat':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'IVF':
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        
        if self.index_type == 'IVF':
            
            for i in range(0, len(self.data), self.batch_size):
                batch = self.data[i:i+self.batch_size].astype(np.float32)
                self.index.train(batch)
        
        
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i+self.batch_size].astype(np.float32)
            self.index.add(batch)
        
        if self.index_type == 'IVF':
            self.index.nprobe = self.nprobe

    def search(self, query, k=5):
        
        if self.index is None:
            raise ValueError("WangHaoyu: Index is not built. Please call build_index() method.")
        
        if isinstance(query, np.ndarray):
            query = query.astype(np.float32)
        
        start_time = time.time()
        distances, indices = self.index.search(query.reshape(1, -1), k)
        end_time = time.time()

        return {
            'indices': indices[0],
            'distances': distances[0],
            'search_time': end_time - start_time
        }

    def batch_search(self, queries, k=5):
        print("WangHaoyu: Start to batch search")
        if self.index is None:
            raise ValueError("WangHaoyu: Index is not built. Please call build_index() method.")
        
        if isinstance(queries, np.ndarray):
            queries = queries.astype(np.float32)
        
        start_time = time.time()
        distances, indices = self.index.search(queries, k)
        end_time = time.time()

        return {
            'indices': indices,
            'distances': distances,
            'search_time': end_time - start_time
        }

    def save_index(self, filename):
        print("WangHaoyu: Start to save index")
        if self.index is None:
            raise ValueError("WangHaoyu: Index is not built. Please call build_index() method.")
        faiss.write_index(self.index, filename)

    def load_index(self, filename):
        
        self.index = faiss.read_index(filename)
        if self.index_type == 'IVF':
            self.index.nprobe = self.nprobe


# data1 = np.load('/mnt/data/why/MAESTRO2004_embedding_result/MAESTRO2004_melody_features.npy')
# data2 = np.load('/mnt/data/why/Musicbench_embedding_result/melody_features_epoch_1.npy')


# combined_data = np.concatenate((data1, data2), axis=0)




# builder = FaissDatasetBuilder(data=combined_data, index_type='IVF', batch_size=10000)


# builder.build_index()



# results = builder.search(query, k=5)








# batch_results = builder.batch_search(batch_queries, k=5)






# builder.save_index("combined_melody_features_index.faiss")