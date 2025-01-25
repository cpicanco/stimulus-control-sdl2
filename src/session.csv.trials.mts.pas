{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.csv.trials.mts;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, session.csv.trials.base;

type

  { TCSVTrialsMTS }

  TCSVTrialsMTS = class(TCSVTrialsBase)
    private
      FSamples       : integer;
      FComparisons   : integer;
      FRelation      : string;
      FHasPrompt     : Boolean;
      FHasTextPrompt : Boolean;
      FPrompt        : string;
      FFontName      : string;
    public
      constructor Create(ASource: string); override;
      property Comparisons : integer read FComparisons write FComparisons;
      property Relation : string read FRelation write FRelation;
      property Samples : integer read FSamples write FSamples;
  end;

implementation

uses
  sdl.app.trials.mts
  , session.constants.mts;

{ TCSVTrialsMTS }

constructor TCSVTrialsMTS.Create(ASource: string);
begin
  inherited Create(ASource);
  FKind := TMTS.ClassName;
  FSamples     := 1;
  FComparisons := 0;
  FRelation    := '';
  FHasPrompt   := False;
  FHasTextPrompt := True;
  FPrompt      := '';
  FFontName    := '';

  with MTSKeys do begin
    RegisterParameter(SamplesKey,
      @FSamples, FSamples);
    RegisterParameter(ComparisonsKey,
      @FComparisons, FComparisons);
    RegisterParameter(RelationKey,
      @FRelation, FRelation);
    RegisterParameter(HasPromptKey,
      @FHasPrompt, FHasPrompt);
    RegisterParameter(HasTextPromptKey,
      @FHasTextPrompt, FHasTextPrompt);
    RegisterParameter(PromptKey,
      @FPrompt, FPrompt);
    RegisterParameter(FontNameKey,
      @FFontName, FFontName);
  end;
end;

end.


