{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit picanco.experiments.words.constants;

{$mode ObjFPC}{$H+}

interface

uses
  picanco.experiments.words.types;

const
  PlosiveBilabial : TConsonant = (
    Ord: csPlosiveBilabial;
    IPA: 'b';
    SYM: 'b';
    HumanReadable: 'b');

  NonSibilantFricative : TConsonant = (
    Ord: csNonSibilantFricative;
    IPA: 'f';
    SYM: 'f';
    HumanReadable: 'f');

  LateralApproximantAlveolar : TConsonant = (
    Ord: csLateralApproximantAlveolar;
    IPA: 'l';
    SYM: 'l';
    HumanReadable: 'l');

  NasalAlveolar : TConsonant = (
    Ord: csNasalAlveolar;
    IPA: 'n';
    SYM: 'n';
    HumanReadable: 'n');

  OpenFront : TVowel = (
    Ord: vsOpenFront;
    IPA: 'a';
    SYM: 'aa';
    HumanReadable: 'a';
    HumanReadableStress: 'á');

  OpenMidFront : TVowel = (
    Ord: vsOpenMidFront;
    IPA: 'ɛ';
    SYM: 'eh';
    HumanReadable: 'e';
    HumanReadableStress: 'é');

  CloseFront : TVowel = (
    Ord: vsCloseFront;
    IPA: 'i';
    SYM: 'ih';
    HumanReadable: 'i';
    HumanReadableStress: 'í');

  OpenMidBack : TVowel = (
    Ord: vsOpenMidBack;
    IPA: 'ɔ';
    SYM: 'ao';
    HumanReadable: 'o';
    HumanReadableStress: 'ó');

implementation

end.

