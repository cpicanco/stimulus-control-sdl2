{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.constants.mts;

{$mode ObjFPC}{$H+}

interface

type
  TMTSKeys = record
    RelationKey    : string;
    SamplesKey     : string;
    SampleKey      : string;
    ComparisonsKey : string;
    ComparisonKey  : string;
    HasPromptKey   : string;
    HasTextPromptKey  : string;
    PromptKey      : string;
    FontNameKey    : string;
  end;

const
  HeaderSample = 'S';
  HeaderComparisons = 'Comparisons';
  HeaderComparison = 'C';
  HeaderRelation = 'Relation';
  HeaderHasPrompt = 'HasPrompt';

  MTSKeys : TMTSKeys = (
    RelationKey    : HeaderRelation;
    SamplesKey     : 'Samples';
    SampleKey      : HeaderSample;
    ComparisonsKey : HeaderComparisons;
    ComparisonKey  : HeaderComparison;
    HasPromptKey   : HeaderHasPrompt;
    HasTextPromptKey  : 'HasTextPrompt';
    PromptKey      : 'Prompt';
    FontNameKey    : 'FontName');

implementation

end.

