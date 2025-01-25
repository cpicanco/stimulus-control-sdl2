{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit experiments.constants.trials.pseudowords;

{$mode ObjFPC}{$H+}

interface


type
  TParserTrialsPseudowords = record
    Cycle       : string;
    Condition   : string;
    Comparisons : string;
    Relation    : string;
    Code        : string;
  end;

const
  ParserTrialsPseudowords : TParserTrialsPseudowords = (
    Cycle       : 'Cycle';
    Condition   : 'Condition';
    Comparisons : 'Comparisons';
    Relation    : 'Relation';
    Code        : 'Code';
  );

implementation

end.

