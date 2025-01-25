{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.constants.trials.dapaap;

{$mode ObjFPC}{$H+}

interface

uses session.constants.mts, session.constants.trials;

type
  TParserTrialsDAPAAP = record
    ReferenceNameKey : string;
    NameKey          : string;
    SampleKey        : string;
    ComparisonKey    : string;
    RelationKey      : string;
    SubsetKey        : string;
    HasPromptKey     : string;
    TotalLoopsKey    : string;
  end;

const
  ParserTrialsDAPAAP : TParserTrialsDAPAAP = (
    ReferenceNameKey : HeaderReferenceName;
    NameKey          : HeaderName;
    SampleKey        : HeaderSample;
    ComparisonKey    : HeaderComparison;
    RelationKey      : HeaderRelation;
    SubsetKey        : 'Subset';
    HasPromptKey     : HeaderHasPrompt;
    TotalLoopsKey    : 'TotalLoops';
  );

implementation

end.

