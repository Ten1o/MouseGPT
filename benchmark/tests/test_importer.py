"""Regression tests for faithful archived evaluation ingestion (stdlib only)."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from xml.sax.saxutils import escape

try:
    from mousebench.importer import import_catalog, import_ratings, read_xlsx, _observation, _model_label
except ModuleNotFoundError:
    # Allows the delivery bundle to be tested before installation into the package.
    spec = importlib.util.spec_from_file_location('importer', Path(__file__).with_name('importer.py'))
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    import_catalog, import_ratings, read_xlsx = importer.import_catalog, importer.import_ratings, importer.read_xlsx
    _observation, _model_label = importer._observation, importer._model_label

SAMPLE = '240420_C10M1_00003972'
HEADERS = ['image_path', 'Overall', 'Head', 'Limb', 'Torso', 'Others', 'Keywords', 'usr', 'model']

def inline(ref, value):
    return '<c r="%s" t="inlineStr"><is><t>%s</t></is></c>' % (ref, escape(str(value)))

def worksheet(rows):
    return '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + ''.join('<row r="%s">%s</row>' % (number, cells) for number, cells in rows) + '</sheetData></worksheet>'

def make_xlsx(path, sheets):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('xl/workbook.xml', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + ''.join('<sheet name="%s" sheetId="%s" r:id="rId%s"/>' % (name, n, n) for n, (name, _) in enumerate(sheets, 1)) + '</sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join('<Relationship Id="rId%s" Target="worksheets/sheet%s.xml"/>' % (n, n) for n in range(1, len(sheets) + 1)) + '</Relationships>')
        archive.writestr('xl/sharedStrings.xml', '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><r><t>reviewer</t></r><r><t>甲</t></r></si></sst>')
        for n, (_, rows) in enumerate(sheets, 1):
            archive.writestr('xl/worksheets/sheet%s.xml' % n, worksheet(rows))

def rubric_sheet():
    header = ''.join(inline(chr(65 + i) + '1', h) for i, h in enumerate(HEADERS))
    data = inline('A2', 'exports/minicpm-2.6_' + SAMPLE + '.jpg')
    data += '<c r="B2" t="b"><v>1</v></c><c r="D2" t="b"><v>0</v></c>'
    data += '<c r="H2" t="s"><v>0</v></c><c r="I2" t="str"><f>LEFT(A2,3)</f><v>minicpm-2.6</v></c>'
    return [(1, header), (2, data), (999, '<c r="I999"><f>LEFT(A999,3)</f></c>')]

class ImporterTests(unittest.TestCase):
    def test_sparse_ooxml_formula_cache_and_duplicate_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sheets = [('original', rubric_sheet()), ('output', rubric_sheet()),
                      ('Rates', [(1, inline('A1', 'Model') + inline('B1', 'Overall (True Rate)')),
                                 (2, inline('A2', 'minicpm-2.6') + '<c r="B2"><v>0.7</v></c>')])]
            make_xlsx(root / 'scores.xlsx', sheets)
            (root / '~$scores.xlsx').write_bytes(b'not a workbook')
            ratings, sources, rejected = import_ratings(root, {SAMPLE})
            self.assertEqual(len(ratings), 2)
            self.assertEqual(rejected, [])
            self.assertEqual(len(sources), 1)
            first, second = ratings
            self.assertNotEqual(first['id'], second['id'])
            self.assertEqual(first['duplicate_fingerprint'], second['duplicate_fingerprint'])
            self.assertEqual(first['matching_observation_copies'], 2)
            self.assertIs(first['value']['Overall'], True)
            self.assertIs(first['value']['Limb'], False)
            self.assertIsNone(first['value']['Head'])
            self.assertEqual(first['reviewer'], 'reviewer甲')
            self.assertEqual(first['source_cells']['I2']['formula'], 'LEFT(A2,3)')
            self.assertEqual(first['source_cells']['I2']['value'], 'minicpm-2.6')
            self.assertEqual(first['linkage_status'], 'frame_only_unverified_output')
            self.assertIsNone(first['candidate_id'])
            self.assertFalse(first['is_ground_truth_caption'])
            again = import_ratings(root, {SAMPLE})[0]
            self.assertEqual([r['id'] for r in ratings], [r['id'] for r in again])
            self.assertEqual(sources[0]['sheets'][2]['kind'], 'derived_or_other')

    def test_pair_csv_corrections_source_hash_and_unknown_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = '240903-dual-production-gpt4o##minicpm-2.6=' + SAMPLE + '_zh-rater'
            for name in ['before.csv', 'correction.csv']:
                with (root / name).open('w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Key', 'Result', 'Time'])
                    writer.writerow([key, 'Both good-2024-10-23T13:26:16.174980', ''])
                    writer.writerow([key.replace(SAMPLE, '240420_C99M9_00000012'), 'Both bad', ''])
            ratings, sources, rejected = import_ratings(root, {SAMPLE})
            self.assertEqual(len(ratings), 2)
            self.assertEqual(len(rejected), 2)
            self.assertEqual({r['reason'] for r in rejected}, {'frame_not_in_audit'})
            self.assertNotEqual(ratings[0]['id'], ratings[1]['id'])
            self.assertEqual(ratings[0]['duplicate_fingerprint'], ratings[1]['duplicate_fingerprint'])
            self.assertEqual(ratings[0]['value'], 'Both good')
            self.assertEqual(ratings[0]['timestamp'], '2024-10-23T13:26:16.174980')
            self.assertEqual(ratings[0]['reviewer'], 'rater')
            self.assertEqual(ratings[0]['model_b'], 'minicpm-2.6')
            self.assertTrue(all(len(source['sha256']) == 64 for source in sources))

    def test_explicit_preference_with_unknown_metadata_is_retained(self):
        observation, reason = _observation('pairwise', {
            'Key': '240903-dual-production-minicpm-2.6##gpt4o=' + SAMPLE + '_zh',
            'output': 'B is better - ?', 'Results': 'B is better ', 'Time': ' ?',
            'Username': '2.6##gpt4o=' + SAMPLE + '_zh'})
        self.assertIsNone(reason)
        self.assertEqual(observation['value'], 'B is better')
        self.assertEqual(observation['reviewer'], '')
        self.assertIn('unknown_or_invalid_timestamp', observation['issues'])
        self.assertIn('invalid_derived_reviewer_cell', observation['issues'])
        self.assertIn('reviewer_unknown', observation['issues'])

    def test_model_label_uses_exact_directory_not_truncated_convenience_label(self):
        self.assertEqual(_model_label({'nas_path': '/output/' + SAMPLE + '/MiniCPM-V-2_6/output_des/00003972.txt',
                                      'path_model_labels_json': '["MiniCPM-V-2"]'}), 'MiniCPM-V-2_6')
        self.assertEqual(_model_label({'nas_path': '/raw_descriptions/00003972.json',
                                      'actual_model_metadata_json': '["gpt-4-turbo-2024-04-09"]',
                                      'path_model_labels_json': '["gpt4o"]'}), 'gpt-4-turbo-2024-04-09')

    def test_catalog_links_assets_by_full_identity_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); audit = root / 'audit'; evaluation = root / 'evaluation'
            audit.mkdir(); evaluation.mkdir()
            frame = dict(sample_id=SAMPLE, capture_date='20240420', group_id='C10M1',
                         frame_id='00003972', recording_id='240420_C10M1', frame_identity='20240420/C10M1/00003972')
            with (audit / 'frame_summary.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(frame)); writer.writeheader(); writer.writerow(frame)
            asset = dict(sample_id=SAMPLE, storage_kind='filesystem_path', nas_path='/volume/test/00003972.jpg',
                         archive_member_path='', record_locator='', asset_role='source_concat',
                         relation_level='same_frame/raw', latest_size='100', frame_membership_verified='True',
                         original_export_identity_verified='False', evaluation_use_verified='False',
                         latest_is_regular_file='True', latest_is_symlink='False', latest_recorded_checked_at='2026-09-27T01:00:00Z')
            with (audit / 'frame_asset_inventory.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(asset)); writer.writeheader(); writer.writerow(asset)
            first = import_catalog(audit, evaluation); second = import_catalog(audit, evaluation)
            self.assertEqual(first, second)
            self.assertEqual(first['frames'][0]['id'], SAMPLE)
            self.assertEqual(first['assets'][0]['frame_id'], SAMPLE)
            self.assertEqual(first['assets'][0]['role'], 'source_concat')
            self.assertEqual(first['assets'][0]['source_row'], 2)
            self.assertFalse(first['assets'][0]['evaluation_use_verified'])
            self.assertEqual(first['summary']['frame_count'], 1)
            self.assertEqual(first['summary']['evaluation_file_count'], 0)

if __name__ == '__main__':
    unittest.main()
