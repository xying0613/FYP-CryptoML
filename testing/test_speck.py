import unittest
import speck as sp
import numpy as np

class TestSpeckPY(unittest.TestCase):
  
    def test_word_size(self):
        self.assertEqual(16, sp.WORD_SIZE())

    def test_alpha(self):
        self.assertEqual(7, sp.ALPHA())

    def test_beta(self):
        self.assertEqual(2, sp.BETA())

    def test_rol(self):
        self.assertEqual(62465, sp.rol(1000,7))
        self.assertEqual(0, sp.rol(0,0))

        with self.assertRaises(TypeError):
            sp.rol(None,0)

        with self.assertRaises(TypeError):
            sp.rol(0,None)

    def test_ror(self):
        self.assertEqual(53255, sp.ror(1000,7))
        self.assertEqual(0, sp.ror(0,0))

        with self.assertRaises(TypeError):
            sp.ror(None,0)

        with self.assertRaises(TypeError):
            sp.ror(0,None)

    def test_enc_one_round(self):
        self.assertEqual((32805, 32933), sp.enc_one_round((0x0040,0x0020),5))
        
        with self.assertRaises(TypeError):
            sp.enc_one_round(0x0040,5)
            
        with self.assertRaises(TypeError):
            sp.enc_one_round((0,0.5),5)

        with self.assertRaises(TypeError):
            sp.enc_one_round((0x0040,0x0020),0.5)

    def test_dec_one_round(self):
        self.assertEqual((0x0040,0x0020), sp.dec_one_round((32805, 32933),5))
        
        with self.assertRaises(TypeError):
            sp.dec_one_round(0x0040,5)
            
        with self.assertRaises(TypeError):
            sp.dec_one_round((0,0.5),5)

        with self.assertRaises(TypeError):
            sp.dec_one_round((32805, 32933),0.5)

    def test_expand_key(self):
        self.assertEqual([3, 1039, 5682], sp.expand_key([1,2,3],3))
        
        # k must be a list
        with self.assertRaises(TypeError):
            sp.expand_key(0,0)

        # k cannot be an empty list
        with self.assertRaises(IndexError):
            sp.expand_key([],0)

        # t cannot be 0
        with self.assertRaises(IndexError):
            sp.expand_key([2,4,6,8,10],0)

        # t cannot be larger than len(k)
        with self.assertRaises(IndexError):
            sp.expand_key([1,2,3],4)
            

    def test_encrypt(self):
        # ks must be a list
        with self.assertRaises(TypeError):
            sp.encrypt((0x0040,0x0020),0)
            
        # p must be subscriptable
        with self.assertRaises(TypeError):
            sp.encrypt(0x0040,5)

        self.assertEqual((2223,3128),sp.encrypt((0x0040,0x0020),[1,2,3]))
        self.assertEqual((32805,32933),sp.encrypt((0x0040,0x0020),[5]))


    def test_decrypt(self):
        # ks must be a list
        with self.assertRaises(TypeError):
            sp.decrypt((32805,32933),0)

        # c must be subscriptable
        with self.assertRaises(TypeError):
            sp.decrypt(32933,5)

        self.assertEqual((0x0040,0x0020),sp.decrypt((2223,3128),[1,2,3]))
        self.assertEqual((0x0040,0x0020),sp.decrypt((32805,32933),[5]))


    def test_convert_to_binary(self):

        arr_2pt = np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3]], dtype=np.uint16)
        arr_4pt = np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],
                            [1,2,3],[1,2,3],[1,2,3],[1,2,3]], dtype=np.uint16)

        with self.assertRaises(AssertionError):
            sp.convert_to_binary(arr_2pt, 1)
            
        with self.assertRaises(AssertionError):
            sp.convert_to_binary(arr_2pt, 4)

        with self.assertRaises(AssertionError):
            sp.convert_to_binary(arr_4pt, 2)

        with self.assertRaises(TypeError):
            sp.convert_to_binary([[1,2,3],[1,2,3],[1,2,3],[1,2,3]], 2)


    def test_make_train_data_2pt(self):

        with self.assertRaises(IndexError):
            sp.make_train_data_2pt(10, 0)

        with self.assertRaises(TypeError):
            sp.make_train_data_2pt(10,1,0)

        self.assertEqual((10,64), sp.make_train_data_2pt(10,1)[0].shape)
        self.assertEqual((10,), sp.make_train_data_2pt(10,1)[1].shape)


    def test_real_differences_data_2pt(self):
        
        with self.assertRaises(IndexError):
            sp.real_differences_data_2pt(10, 0)

        with self.assertRaises(TypeError):
            sp.real_differences_data_2pt(10,1,0)

        self.assertEqual((10,64), sp.real_differences_data_2pt(10,1)[0].shape)
        self.assertEqual((10,), sp.real_differences_data_2pt(10,1)[1].shape)


    def test_make_train_data_4pt(self):

        with self.assertRaises(IndexError):
            sp.make_train_data_4pt(10, 0)

        with self.assertRaises(TypeError):
            sp.make_train_data_4pt(10,1,0,0)

        self.assertEqual((10,128), sp.make_train_data_4pt(10,1)[0].shape)
        self.assertEqual((10,), sp.make_train_data_4pt(10,1)[1].shape)


    def test_real_differences_data_4pt(self):
        
        with self.assertRaises(IndexError):
            sp.real_differences_data_4pt(10, 0)

        with self.assertRaises(TypeError):
            sp.real_differences_data_4pt(10,1,0,0)

        self.assertEqual((10,128), sp.real_differences_data_4pt(10,1)[0].shape)
        self.assertEqual((10,), sp.real_differences_data_4pt(10,1)[1].shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)
    
