{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.constants.trials.pseudowords;

{$mode ObjFPC}{$H+}

interface

uses session.constants.mts, session.constants.trials;

type

  TParserTrialsPseudowordsMTS = record
    ReferenceNameKey : string;
    NameKey          : string;
    SampleKey        : string;
    ComparisonsKey   : string;
    ComparisonKey    : string;
    RelationKey      : string;
    CycleKey         : string;
    ConditionKey     : string;
    CodeKey          : string;
  end;

const
  ParserTrialsPseudowordsMTS : TParserTrialsPseudowordsMTS = (
    ReferenceNameKey : HeaderReferenceName;
    NameKey          : HeaderName;
    SampleKey        : HeaderSample;
    ComparisonsKey   : HeaderComparisons;
    ComparisonKey    : HeaderComparison;
    RelationKey      : HeaderRelation;
    CycleKey         : 'Cycle';
    ConditionKey     : 'Condition';
    CodeKey          : 'Code';
  );

implementation

end.

