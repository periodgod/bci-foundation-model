"""下载器的清单与续传约定。

这些断言对应两个真实踩过的坑：
1. Sleep-EDF 的 RECORDS 只列 PSG，标注文件（Hypnogram）不在里面，且命名不能靠替换推导；
2. 中断的区间下载会留下"大小正确、内容是空洞"的 .part，续传位置越过末尾会拿到 416。
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_corpora as downloader  # noqa: E402


SUM = """\
aaa sleep-cassette/SC4001E0-PSG.edf
bbb sleep-cassette/SC4001EC-Hypnogram.edf
ccc sleep-cassette/SC4011E0-PSG.edf
ddd sleep-cassette/SC4011EH-Hypnogram.edf
eee sleep-telemetry/ST7011J0-PSG.edf
fff sleep-telemetry/ST7011JO-Hypnogram.edf
"""


class CompanionFiles(unittest.TestCase):
    def setUp(self):
        self.checksums = downloader.parse_checksums(SUM)

    def test_pairs_every_psg_with_its_own_hypnogram(self):
        records = ["sleep-cassette/SC4001E0-PSG.edf",
                   "sleep-cassette/SC4011E0-PSG.edf",
                   "sleep-telemetry/ST7011J0-PSG.edf"]
        self.assertEqual(downloader.companion_files(records, self.checksums), [
            "sleep-cassette/SC4001EC-Hypnogram.edf",
            "sleep-cassette/SC4011EH-Hypnogram.edf",
            "sleep-telemetry/ST7011JO-Hypnogram.edf",
        ])

    def test_version_letter_varies_per_record(self):
        # SC4001E0 对 EC，SC4011E0 对 EH——同一个库里两种写法，字符串替换会错。
        records = ["sleep-cassette/SC4001E0-PSG.edf", "sleep-cassette/SC4011E0-PSG.edf"]
        paired = downloader.companion_files(records, self.checksums)
        self.assertNotEqual(paired[0][-16:], "EH-Hypnogram.edf")
        self.assertEqual(paired[1][-16:], "EH-Hypnogram.edf")

    def test_missing_companion_is_an_error_not_a_guess(self):
        records = ["sleep-cassette/SC9999E0-PSG.edf"]
        with self.assertRaises(RuntimeError):
            downloader.companion_files(records, self.checksums)

    def test_ambiguous_prefix_is_rejected(self):
        checksums = dict(self.checksums)
        # parse_checksums 以路径为键：同一前缀下塞进第二个标注文件。
        checksums["sleep-cassette/SC4001EX-Hypnogram.edf"] = "zzz"
        with self.assertRaises(RuntimeError):
            downloader.companion_files(["sleep-cassette/SC4001E0-PSG.edf"], checksums)


class StalePartial(unittest.TestCase):
    def test_full_size_partial_is_restarted_not_appended(self):
        """大小已等于远端长度但内容不匹配的 .part 必须删掉重下。

        真实故障：区间模式把 .part 预分配成整文件大小后中断，续传位置 == 长度，
        Range: bytes=N- 直接返回 416，整批 11 个文件全部失败。
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = "sleep-cassette/SC0001E0-PSG.edf"
            target = root / path
            target.parent.mkdir(parents=True)
            part = target.with_suffix(target.suffix + ".part")
            part.write_bytes(b"\0" * 1000)

            real_remote_size = downloader.remote_size
            real_open = downloader._open

            downloader.remote_size = lambda *a, **k: 1000

            class Response:
                status = 200

                def read(self, size=-1):
                    return b"" if getattr(self, "used", False) else self._body()

                def _body(self):
                    self.used = True
                    return b"payload"

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            def fake_open(url, offset=0, timeout=60):
                # 旧实现会带着 offset=1000 去请求，这里显式断言已从头开始。
                assert offset == 0, f"续传位置应重置为 0，实际 {offset}"
                return Response()

            downloader._open = fake_open
            try:
                row = downloader.download_one("sleep-edfx", "1.0.0", path, root, None)
            finally:
                downloader.remote_size = real_remote_size
                downloader._open = real_open
            self.assertEqual(row["status"], "downloaded")
            self.assertEqual(target.read_bytes(), b"payload")


class ScanSleepPairs(unittest.TestCase):
    """ingest 侧的目录扫描必须覆盖两个子库，且认得出非常规版本位。"""

    def test_scans_both_sub_libraries(self):
        from eeg_mime_foundation.pairing import scan_sleep_pairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub, names in (("sleep-cassette", [("SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"),
                                                   ("SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf")]),
                               ("sleep-telemetry", [("ST7011J0-PSG.edf", "ST7011JO-Hypnogram.edf")])):
                (root / sub).mkdir()
                for psg, hyp in names:
                    (root / sub / psg).write_bytes(b"")
                    (root / sub / hyp).write_bytes(b"")
            pairs = scan_sleep_pairs(root)
            self.assertEqual(len(pairs), 3)
            self.assertIn(("sleep-telemetry/ST7011J0-PSG.edf",
                           "sleep-telemetry/ST7011JO-Hypnogram.edf"), pairs)

    def test_missing_annotation_raises_instead_of_silently_skipping(self):
        from eeg_mime_foundation.pairing import scan_sleep_pairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sleep-cassette"
            root.mkdir(parents=True)
            (root / "SC4001E0-PSG.edf").write_bytes(b"")
            with self.assertRaises(ValueError):
                scan_sleep_pairs(root.parent)


if __name__ == "__main__":
    unittest.main()
