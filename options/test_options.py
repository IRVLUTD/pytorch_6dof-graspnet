from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--which_epoch',
            type=str,
            default='latest',
            help='which epoch to load? set to latest to use latest cached model'
        )

        self.parser.add_argument(
            '--split_mode',
            type=str,
            default='o',
            help="split_mode: Choose between held-out instance ('i'), tasks ('t') and classes ('o')")
        self.parser.add_argument(
            '--split_version',
            type=int,
            default=1,
            help="0 is for random splits, 1 is for cross-validation splits")
        self.parser.add_argument(
            '--split_idx',
            type=int,
            default=1,
            help="For each split mode, there are 4 cross validation splits (choose between 0-3)"
        )
        self.parser.add_argument('--phase',
                                 type=str,
                                 default='val',
                                 help='train, val, test, etc')
        self.parser.add_argument(
            '--vis_test',
            action='store_true',
            help='visuialize pc and grasp'
        )
        

        self.is_train = False