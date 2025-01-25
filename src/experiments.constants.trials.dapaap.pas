{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit experiments.constants.trials.dapaap;

{$mode ObjFPC}{$H+}

interface


type
  TParserTrialsDAPAAP = record
    Subset     : string;
    Condition  : string;
    Comparison : string;
    Relation   : string;
    Sample     : string;
  end;

const
  ParserTrialsDAPAAP : TParserTrialsDAPAAP = (
    Subset     : 'Subset';
    Condition  : 'Condition';
    Comparison : 'Word2';
    Relation   : 'Relation';
    Sample     : 'Word1';
  );

implementation

end.

